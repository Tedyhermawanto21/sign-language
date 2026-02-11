# sign2speech_final.py
# Webcam -> MediaPipe keypoints -> Transformer Encoder + CTC -> Teks Kalimat -> TTS
# Modes: collect / train / infer

import os, glob, uuid, argparse, time, threading
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.backend import ctc_decode, ctc_batch_cost
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Konfigurasi Utama ======
DATA_PATH       = "MP_Data_Sentences"
SEQUENCE_LENGTH = 60               # samakan untuk collect/train/infer
FPS_TARGET      = 24
VOCAB_STR       = " abcdefghijklmnopqrstuvwxyz0123456789.,?!'-"  # spasi + karakter
NUM_CLASSES     = len(VOCAB_STR) + 1  # +1 utk CTC blank
FEATURE_DIM     = 1662            # pose(33*4)=132 + face(468*3)=1404 + hands(21*3*2)=126
MODEL_DIR       = "models"
MODEL_WEIGHTS   = os.path.join(MODEL_DIR, "sign2speech_ctc.h5")

# --- Drawing flags ---
DRAW_FACE_MODE = "none"       # "contours" | "points" | "tesselation" | "none"
DRAW_POSE = True              # True: tampilkan pose landmarks (bahu/lengan penting untuk sign language)
FLIP_CAMERA = True            # True: flip horizontal agar tidak mirror (kiri=kiri)

# --- Fitur wajah (disarankan False untuk dataset kecil) ---
USE_FACE = False                   # True: pakai face di infer; False: di-nolkan
ZERO_FACE_ON_LOAD = True           # True: nolkan face saat TRAIN (butuh retrain)
POSE_SLICE = slice(0, 132)            # 33*4
FACE_SLICE = slice(132, 1536)         # 468*3
LH_SLICE   = slice(1536, 1599)        # 21*3
RH_SLICE   = slice(1599, 1662)        # 21*3

HAND_VIS_TH = 0.15  # minimal 15% frame mendeteksi tangan agar boleh commit

# --- Commit & smoothing ---
CONF_TH = 0.45                     # ambang confidence rata-rata softmax
IDLE_FACTOR = 1.2                  # detik idle = IDLE_FACTOR * FPS_TARGET
BEAM_WIDTH = 20
TOP_PATHS  = 3

MAX_CER_TO_MAP = 0.35   # map ke kamus hanya jika jarak relatif <= 35%
MARGIN_TIE     = 0.10   # kalau kandidat teratas & kedua beda <10%, anggap ambigu

CONTENT_TH = 0.35            # ambang keyakinan "bukan blank"
BLANK_INDEX = len(VOCAB_STR) # indeks token blank utk CTC

# ====== Vocab util ======
char2idx = {ch: i for i, ch in enumerate(VOCAB_STR)}
idx2char = {i: ch for ch, i in char2idx.items()}

def text_to_ints(text: str) -> np.ndarray:
    return np.array([char2idx[c] for c in text.lower() if c in char2idx], dtype=np.int32)

def ints_to_text(ids) -> str:
    return ''.join(idx2char.get(int(i), '') for i in ids)

# ====== MediaPipe setup ======
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_face  = mp.solutions.face_mesh
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands

# --- Hybrid Tracking Config ---
USE_HYBRID_TRACKING = True    # True: Use Hands+Pose (better for 2 hands), False: legacy Holistic
HYBRID_DETECTION_CONF = 0.5   # Detection confidence (lebih rendah = lebih sensitif untuk tangan mengepal)
HYBRID_TRACKING_CONF = 0.4    # Tracking confidence (lebih rendah = tracking lebih persistent)
TEMPORAL_WEIGHT = 0.9         # Weight for temporal position matching (lebih tinggi = identity lebih "lengket")
USE_POSE_WRIST_FALLBACK = True  # Fallback ke Pose wrist landmarks jika Hands tidak mendeteksi

class HybridResults:
    """
    Compatibility wrapper untuk menyimpan hasil dari hybrid tracking
    dengan format mirip mp_holistic.Holistic results.
    """
    def __init__(self):
        self.pose_landmarks = None
        self.face_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.hand_identity_confidence = 0.0  # Confidence of L/R assignment
        self.is_fallback = False  # True jika menggunakan Pose wrist fallback
        
class HybridHandTracker:
    """
    Hybrid hand tracking dengan MediaPipe Hands + Pose + temporal identity tracking.
    Mengatasi masalah hand swap ketika 2 tangan bersinggungan atau overlapping.
    
    Menggunakan Hungarian algorithm untuk optimal matching berdasarkan:
    1. MediaPipe handedness classification
    2. Temporal position continuity (posisi wrist frame sebelumnya)
    """
    
    def __init__(self, max_hands=2, 
                 min_detection_conf=HYBRID_DETECTION_CONF,
                 min_tracking_conf=HYBRID_TRACKING_CONF):
        # MediaPipe Hands - dedicated hand model (lebih akurat untuk multi-hand)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            model_complexity=0  # 0=lite (fastest), 1=full (more accurate)
        )
        # MediaPipe Pose - untuk body landmarks
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # 0=lite (fastest)
        )
        
        # Temporal tracking state
        self.prev_left_wrist = None   # (x, y) normalized
        self.prev_right_wrist = None  # (x, y) normalized
        self.identity_confidence = 0.0
        self.frame_count = 0
        
    def _get_wrist_pos(self, hand_landmarks):
        """Extract wrist position (landmark 0) dari hand landmarks."""
        if hand_landmarks is None:
            return None
        wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0
        return np.array([wrist.x, wrist.y])
    
    def _distance(self, p1, p2):
        """Euclidean distance between two points."""
        if p1 is None or p2 is None:
            return float('inf')
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def _match_hands_to_identity(self, detected_hands, handedness_list):
        """
        Match detected hands ke left/right identity menggunakan kombinasi:
        1. MediaPipe handedness classification
        2. Temporal position continuity
        
        Returns: (left_hand_landmarks, right_hand_landmarks, confidence)
        """
        if not detected_hands:
            # No hands detected - reset temporal state gradually
            if self.frame_count > 10:
                self.prev_left_wrist = None
                self.prev_right_wrist = None
            return None, None, 0.0
        
        n_hands = len(detected_hands)
        
        if n_hands == 1:
            # Single hand - use handedness + temporal info
            hand = detected_hands[0]
            handedness = handedness_list[0].classification[0]
            label = handedness.label  # "Left" or "Right"
            mp_conf = handedness.score
            
            wrist_pos = self._get_wrist_pos(hand)
            
            # Check temporal continuity
            dist_to_prev_left = self._distance(wrist_pos, self.prev_left_wrist)
            dist_to_prev_right = self._distance(wrist_pos, self.prev_right_wrist)
            
            # Combine MediaPipe classification with temporal info
            if self.prev_left_wrist is not None or self.prev_right_wrist is not None:
                # We have temporal info - use weighted decision
                if dist_to_prev_left < dist_to_prev_right:
                    temporal_vote = "Left"
                    temporal_conf = 1.0 / (1.0 + dist_to_prev_left * 10)
                else:
                    temporal_vote = "Right"
                    temporal_conf = 1.0 / (1.0 + dist_to_prev_right * 10)
                
                # Weighted combination
                if temporal_vote == label:
                    final_label = label
                    confidence = TEMPORAL_WEIGHT * temporal_conf + (1-TEMPORAL_WEIGHT) * mp_conf
                else:
                    # Conflict! Use temporal if strong, else use MP
                    if temporal_conf > 0.6 and dist_to_prev_left < 0.15:
                        final_label = temporal_vote
                        confidence = temporal_conf * 0.8
                    else:
                        final_label = label
                        confidence = mp_conf * 0.7
            else:
                # No temporal info - trust MediaPipe
                final_label = label
                confidence = mp_conf
            
            # Update temporal state
            if final_label == "Left":
                self.prev_left_wrist = wrist_pos
                return hand, None, confidence
            else:
                self.prev_right_wrist = wrist_pos
                return None, hand, confidence
        
        else:  # n_hands == 2
            # Two hands - need to assign correctly
            hand0 = detected_hands[0]
            hand1 = detected_hands[1]
            wrist0 = self._get_wrist_pos(hand0)
            wrist1 = self._get_wrist_pos(hand1)
            
            # Get MediaPipe handedness
            h0_label = handedness_list[0].classification[0].label
            h0_conf = handedness_list[0].classification[0].score
            h1_label = handedness_list[1].classification[0].label
            h1_conf = handedness_list[1].classification[0].score
            
            # Build cost matrix for Hungarian algorithm
            # Rows: detected hands [0, 1]
            # Cols: identities [Left, Right]
            cost_matrix = np.zeros((2, 2))
            
            # MediaPipe handedness cost (lower = better match)
            cost_matrix[0, 0] = 0.0 if h0_label == "Left" else 1.0 - h0_conf
            cost_matrix[0, 1] = 0.0 if h0_label == "Right" else 1.0 - h0_conf
            cost_matrix[1, 0] = 0.0 if h1_label == "Left" else 1.0 - h1_conf
            cost_matrix[1, 1] = 0.0 if h1_label == "Right" else 1.0 - h1_conf
            
            # Add temporal cost if we have previous positions
            if self.prev_left_wrist is not None:
                cost_matrix[0, 0] += TEMPORAL_WEIGHT * self._distance(wrist0, self.prev_left_wrist)
                cost_matrix[1, 0] += TEMPORAL_WEIGHT * self._distance(wrist1, self.prev_left_wrist)
            if self.prev_right_wrist is not None:
                cost_matrix[0, 1] += TEMPORAL_WEIGHT * self._distance(wrist0, self.prev_right_wrist)
                cost_matrix[1, 1] += TEMPORAL_WEIGHT * self._distance(wrist1, self.prev_right_wrist)
            
            # Solve assignment (simple 2x2 case - no need for scipy)
            # Compare assignment [0->L, 1->R] vs [0->R, 1->L]
            cost_01 = cost_matrix[0, 0] + cost_matrix[1, 1]  # hand0=Left, hand1=Right
            cost_10 = cost_matrix[0, 1] + cost_matrix[1, 0]  # hand0=Right, hand1=Left
            
            if cost_01 <= cost_10:
                left_hand = hand0
                right_hand = hand1
                self.prev_left_wrist = wrist0
                self.prev_right_wrist = wrist1
            else:
                left_hand = hand1
                right_hand = hand0
                self.prev_left_wrist = wrist1
                self.prev_right_wrist = wrist0
            
            # Confidence based on cost difference
            cost_diff = abs(cost_01 - cost_10)
            confidence = min(1.0, 0.5 + cost_diff)
            
            return left_hand, right_hand, confidence
    
    def _create_synthetic_hand_from_wrist(self, wrist_landmark):
        """
        Create synthetic hand landmarks dari pose wrist landmark.
        Ini adalah fallback ketika Hands model tidak bisa mendeteksi tangan
        (misalnya saat tangan mengepal/terlipat).
        
        Returns: object dengan attribute 'landmark' berisi list of 21 landmarks
        """
        class SyntheticLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        class SyntheticHand:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        # Create 21 landmarks, semua centered di wrist
        # Dengan sedikit offset untuk finger tips agar ada "bentuk"
        landmarks = []
        base_x = wrist_landmark.x
        base_y = wrist_landmark.y
        base_z = wrist_landmark.z if hasattr(wrist_landmark, 'z') else 0.0
        
        # Landmark indices untuk tangan MediaPipe:
        # 0: Wrist
        # 1-4: Thumb (CMC, MCP, IP, TIP)
        # 5-8: Index (MCP, PIP, DIP, TIP)
        # 9-12: Middle (MCP, PIP, DIP, TIP)
        # 13-16: Ring (MCP, PIP, DIP, TIP)
        # 17-20: Pinky (MCP, PIP, DIP, TIP)
        
        # Offset patterns untuk simulasi pose "mengepal"
        finger_offsets = [
            (0, 0),      # 0: Wrist
            (-0.02, -0.02), (-0.03, -0.03), (-0.03, -0.02), (-0.02, -0.01),  # Thumb
            (-0.01, -0.04), (-0.01, -0.03), (-0.01, -0.02), (-0.01, -0.01),  # Index
            (0, -0.04), (0, -0.03), (0, -0.02), (0, -0.01),                   # Middle
            (0.01, -0.04), (0.01, -0.03), (0.01, -0.02), (0.01, -0.01),       # Ring
            (0.02, -0.03), (0.02, -0.02), (0.02, -0.015), (0.02, -0.01),      # Pinky
        ]
        
        for dx, dy in finger_offsets:
            landmarks.append(SyntheticLandmark(
                x=base_x + dx,
                y=base_y + dy,
                z=base_z
            ))
        
        return SyntheticHand(landmarks)
    
    def _get_pose_wrists(self, pose_landmarks):
        """
        Extract left and right wrist landmarks dari Pose.
        Pose landmark indices: 15 = left wrist, 16 = right wrist
        """
        if pose_landmarks is None:
            return None, None
        
        left_wrist = pose_landmarks.landmark[15]   # Left wrist
        right_wrist = pose_landmarks.landmark[16]  # Right wrist
        
        # Check visibility
        left_vis = left_wrist.visibility if hasattr(left_wrist, 'visibility') else 1.0
        right_vis = right_wrist.visibility if hasattr(right_wrist, 'visibility') else 1.0
        
        left = left_wrist if left_vis > 0.3 else None
        right = right_wrist if right_vis > 0.3 else None
        
        return left, right
    
    def process(self, image_rgb):
        """
        Process frame dan return HybridResults dengan consistent hand identity.
        
        Args:
            image_rgb: RGB image (sudah di-flip jika perlu)
            
        Returns:
            HybridResults object dengan pose_landmarks, left_hand_landmarks, 
            right_hand_landmarks, dan hand_identity_confidence
        """
        self.frame_count += 1
        results = HybridResults()
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        results.pose_landmarks = pose_results.pose_landmarks
        
        # Process hands
        hands_results = self.hands.process(image_rgb)
        
        hands_detected = hands_results.multi_hand_landmarks and hands_results.multi_handedness
        
        if hands_detected:
            left_hand, right_hand, confidence = self._match_hands_to_identity(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            )
            results.left_hand_landmarks = left_hand
            results.right_hand_landmarks = right_hand
            results.hand_identity_confidence = confidence
            self.identity_confidence = confidence
        elif USE_POSE_WRIST_FALLBACK and pose_results.pose_landmarks:
            # FALLBACK: Use Pose wrist landmarks untuk create synthetic hands
            left_wrist, right_wrist = self._get_pose_wrists(pose_results.pose_landmarks)
            
            if left_wrist is not None:
                results.left_hand_landmarks = self._create_synthetic_hand_from_wrist(left_wrist)
                self.prev_left_wrist = np.array([left_wrist.x, left_wrist.y])
            if right_wrist is not None:
                results.right_hand_landmarks = self._create_synthetic_hand_from_wrist(right_wrist)
                self.prev_right_wrist = np.array([right_wrist.x, right_wrist.y])
            
            # Lower confidence untuk fallback
            results.hand_identity_confidence = 0.5
            self.identity_confidence = 0.5
            results.is_fallback = True  # Flag untuk indicate ini fallback
        else:
            # No hands and no fallback - keep previous confidence decaying
            self.identity_confidence *= 0.9
            results.hand_identity_confidence = self.identity_confidence
        
        return results
    
    def close(self):
        """Release resources."""
        self.hands.close()
        self.pose.close()

# Global hybrid tracker instance (akan di-init saat pertama kali dipakai)
_hybrid_tracker = None

def get_hybrid_tracker():
    """Get or create global hybrid tracker instance."""
    global _hybrid_tracker
    if _hybrid_tracker is None:
        _hybrid_tracker = HybridHandTracker()
    return _hybrid_tracker

def extract_keypoints(results):
    # POSE (33x4)
    pose = np.zeros((33, 4), dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array([[r.x, r.y, r.z, r.visibility]
                        for r in results.pose_landmarks.landmark], dtype=np.float32)

    # HANDS (21x3)
    lh = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        lh = np.array([[r.x, r.y, r.z]
                      for r in results.left_hand_landmarks.landmark], dtype=np.float32)

    rh = np.zeros((21, 3), dtype=np.float32)
    if results.right_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z]
                      for r in results.right_hand_landmarks.landmark], dtype=np.float32)

    # FACE (468x3)
    face = np.zeros((468, 3), dtype=np.float32)
    if USE_FACE and results.face_landmarks:
        face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark], dtype=np.float32)

    # flatten (RAW)
    return np.concatenate([
        pose.reshape(-1),
        face.reshape(-1),
        lh.reshape(-1),
        rh.reshape(-1)
    ]).astype(np.float32)

def _normalize_frame(v: np.ndarray, use_face: bool) -> np.ndarray:
    """Root ke tengah bahu & scale dg jarak bahu. v: (1662,) RAW."""
    v = v.astype(np.float32).copy()

    pose = v[POSE_SLICE].reshape(33, 4)
    if pose.sum() == 0:     # pose hilang -> biarkan apa adanya
        return v

    L = pose[11, :3]; R = pose[12, :3]
    scale = float(np.linalg.norm(R - L))
    if scale < 1e-6:
        return v
    center = (L + R) / 2.0

    # pose: hanya x,y,z (kolom 0:3) yang dinormalisasi
    pose[:, :3] = (pose[:, :3] - center) / scale
    v[POSE_SLICE] = pose.reshape(-1)

    # hands
    lh = v[LH_SLICE].reshape(21, 3); rh = v[RH_SLICE].reshape(21, 3)
    lh = (lh - center) / scale;      rh = (rh - center) / scale
    v[LH_SLICE] = lh.reshape(-1);    v[RH_SLICE] = rh.reshape(-1)

    # face (opsional)
    if use_face:
        face = v[FACE_SLICE].reshape(468, 3)
        face = (face - center) / scale
        v[FACE_SLICE] = face.reshape(-1)

    return v

def normalize_batch(X: np.ndarray, use_face: bool) -> np.ndarray:
    """X: (T,1662) RAW -> (T,1662) normalized"""
    return np.stack([_normalize_frame(x, use_face) for x in X], axis=0)

def mediapipe_detection(image, model):
    """
    Process image dengan MediaPipe.
    
    Args:
        image: BGR image dari OpenCV
        model: MediaPipe Holistic model (atau None jika USE_HYBRID_TRACKING=True)
        
    Returns:
        (processed_image, results) - results bisa HybridResults atau Holistic results
    """
    # Flip horizontal jika FLIP_CAMERA aktif (agar kiri=kiri, bukan mirror)
    if FLIP_CAMERA:
        image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    if USE_HYBRID_TRACKING:
        # Use hybrid tracker (Hands + Pose with temporal identity)
        tracker = get_hybrid_tracker()
        results = tracker.process(image_rgb)
    else:
        # Legacy: use Holistic
        results = model.process(image_rgb)
    
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_landmarks(image, results):
    """
    Draw landmarks pada image dengan visual indicators untuk hand identity.
    """
    # FACE
    if hasattr(results, 'face_landmarks') and results.face_landmarks and DRAW_FACE_MODE != "none":
        if DRAW_FACE_MODE == "contours":
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_face.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0),
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0)
            )
        elif DRAW_FACE_MODE == "points":
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, connections=None,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1)
            )
        elif DRAW_FACE_MODE == "tesselation":
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_face.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=0),
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=0)
            )

    # POSE (Upper Body Only) - visualisasi bahu dan lengan penting untuk sign language
    if DRAW_POSE and results.pose_landmarks:
        # Define connection list for upper body only
        upper_body_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)
        ]
        
        # Draw connections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, 
            connections=upper_body_connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    
    # HANDS dengan warna konsisten untuk identity
    # LEFT HAND = Ungu/Magenta
    if results.left_hand_landmarks:
        h, w = image.shape[:2]
        is_synthetic = hasattr(results, 'is_fallback') and results.is_fallback
        
        if is_synthetic:
            # For synthetic (fallback) landmarks, draw simple markers
            for lm in results.left_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 4, (121, 44, 250), -1)
        else:
            # Normal hand - use MediaPipe drawing
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        
        # Add "L" label near wrist
        if results.left_hand_landmarks.landmark:
            wrist = results.left_hand_landmarks.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            label = "L*" if is_synthetic else "L"
            cv2.putText(image, label, (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (121,44,250), 2)
    
    # RIGHT HAND = Oranye
    if results.right_hand_landmarks:
        h, w = image.shape[:2]
        is_synthetic = hasattr(results, 'is_fallback') and results.is_fallback
        
        if is_synthetic:
            # For synthetic (fallback) landmarks, draw simple markers
            for lm in results.right_hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 4, (245, 117, 66), -1)
        else:
            # Normal hand - use MediaPipe drawing
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Add "R" label near wrist
        if results.right_hand_landmarks.landmark:
            wrist = results.right_hand_landmarks.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            label = "R*" if is_synthetic else "R"
            cv2.putText(image, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245,117,66), 2)
    
    # Hand identity confidence indicator (only for hybrid tracking)
    if USE_HYBRID_TRACKING and hasattr(results, 'hand_identity_confidence'):
        conf = results.hand_identity_confidence
        is_fallback = hasattr(results, 'is_fallback') and results.is_fallback
        
        # Draw confidence bar at bottom right
        h, w = image.shape[:2]
        bar_width = 100
        bar_height = 8
        x_start = w - bar_width - 10
        y_start = h - 25
        
        # Background
        cv2.rectangle(image, (x_start, y_start), (x_start + bar_width, y_start + bar_height), (50, 50, 50), -1)
        # Confidence fill
        fill_width = int(bar_width * conf)
        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.rectangle(image, (x_start, y_start), (x_start + fill_width, y_start + bar_height), color, -1)
        
        # Label with fallback indicator
        label = "FB" if is_fallback else "ID"
        cv2.putText(image, f"{label}:{conf:.2f}", (x_start, y_start - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Fallback mode indicator
        if is_fallback:
            cv2.putText(image, "POSE FALLBACK", (x_start - 80, y_start + 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

# ====== Mode 1: Rekam Dataset ======
def collect_dataset(sentences, sequences_per_sentence=20):
    global use_spell_mode
    os.makedirs(DATA_PATH, exist_ok=True)
    for s in sentences:
        os.makedirs(os.path.join(DATA_PATH, s), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    # Setup model berdasarkan mode tracking
    if USE_HYBRID_TRACKING:
        holistic = None  # Will use global hybrid tracker via mediapipe_detection
        print("📍 Using HYBRID tracking (Hands + Pose with temporal identity)")
    else:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("📍 Using LEGACY holistic tracking")
    
    try:
        for sent in sentences:
            for seq in range(sequences_per_sentence):
                seq_dir = os.path.join(DATA_PATH, sent, str(seq))
                os.makedirs(seq_dir, exist_ok=True)

                # tunggu 'r'
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    img, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(img, results)
                    cv2.putText(img, f"[{sent}] Seq {seq} | Tekan 'r' untuk REKAM {SEQUENCE_LENGTH} frame",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.imshow("Collect", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('e'):
                        use_spell_mode = not use_spell_mode  # toggle
                    if key == ord('r'): break
                    if key == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return

                # rekam
                for f in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: break
                    img, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(img, results)
                    kpts = extract_keypoints(results)
                    np.save(os.path.join(seq_dir, f"{f}.npy"), kpts)
                    cv2.putText(img, f"REKAM: {sent} | Seq {seq} | Frame {f+1}/{SEQUENCE_LENGTH}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.imshow("Collect", img)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return

                with open(os.path.join(seq_dir, "label.txt"), "w", encoding="utf-8") as fp:
                    fp.write(sent)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if holistic is not None:
            holistic.close()

def collect_batch(label, target_samples=50, gap_frames=5, delay_sec=3, duration_sec=10):
    """
    CONTINUOUS data collection - Press R once, records until done!
    
    CARA PAKAI:
    - Tekan 'R' untuk mulai (ada countdown)
    - Recording TERUS MENERUS sampai target tercapai
    - Cukup gerak-gerak aja, samples tersimpan otomatis
    - Tekan 'S' untuk pause, 'R' untuk lanjut
    - Tekan 'Q' untuk keluar
    
    Args:
        label: nama kelas/frasa yang direkam
        target_samples: jumlah sample yang ingin direkam
        gap_frames: overlap frame antar sequence (default 5)
        delay_sec: delay countdown sebelum mulai (default 3)
        duration_sec: (tidak dipakai di mode ini)
    """
    import time
    
    os.makedirs(DATA_PATH, exist_ok=True)
    label_dir = os.path.join(DATA_PATH, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Cari sequence number terakhir
    existing = [int(d) for d in os.listdir(label_dir) if d.isdigit()]
    start_seq = max(existing) + 1 if existing else 0
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n{'='*50}")
    print(f"🎬 CONTINUOUS COLLECTION MODE")
    print(f"{'='*50}")
    print(f"Label: {label}")
    print(f"Target: {target_samples} samples")
    print(f"Existing: {len(existing)} samples")
    if USE_HYBRID_TRACKING:
        print(f"🔷 Tracking: HYBRID (Hands + Pose with temporal identity)")
    else:
        print(f"🔶 Tracking: LEGACY Holistic")
    print(f"{'='*50}")
    print("CONTROLS:")
    print("  R = Mulai/Lanjut recording")
    print("  S = Pause recording")
    print("  Q = Keluar")
    print(f"{'='*50}")
    print("Tekan R sekali, lalu gerak-gerak sampai selesai!")
    print(f"{'='*50}\n")
    
    samples_recorded = 0
    current_seq = start_seq
    
    # States: 'idle', 'countdown', 'recording'
    state = 'idle'
    countdown_start = 0
    frame_buffer = []
    recording_start_time = 0
    
    # Setup model berdasarkan mode tracking
    if USE_HYBRID_TRACKING:
        holistic = None  # Will use global hybrid tracker via mediapipe_detection
    else:
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    try:
        while samples_recorded < target_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            img, results = mediapipe_detection(frame, holistic)
            draw_landmarks(img, results)
            
            # Check hand visibility
            left_hand = results.left_hand_landmarks is not None
            right_hand = results.right_hand_landmarks is not None
            hands_visible = left_hand or right_hand
            
            current_time = time.time()
            
            # State machine
            if state == 'countdown':
                elapsed = current_time - countdown_start
                remaining = delay_sec - elapsed
                if remaining <= 0:
                    state = 'recording'
                    frame_buffer = []
                    recording_start_time = time.time()
                    print("🔴 Recording started! Keep moving...")
                else:
                    # Countdown display
                    bar_color = (0, 165, 255)  # Orange
                    cv2.rectangle(img, (0, 0), (img.shape[1], 120), bar_color, -1)
                    cv2.putText(img, f"GET READY!", (img.shape[1]//2 - 100, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(img, f"{int(remaining) + 1}", (img.shape[1]//2 - 20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            
            elif state == 'recording':
                # Recording display - CONTINUOUS until target reached
                elapsed_total = current_time - recording_start_time
                bar_color = (0, 0, 255)  # Red
                cv2.rectangle(img, (0, 0), (img.shape[1], 100), bar_color, -1)
                cv2.putText(img, f"RECORDING... {elapsed_total:.1f}s", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(img, f"Label: {label}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Progress bar (samples progress)
                progress = samples_recorded / target_samples
                bar_width = int((img.shape[1] - 20) * progress)
                cv2.rectangle(img, (10, 75), (img.shape[1] - 10, 90), (100, 100, 100), -1)
                cv2.rectangle(img, (10, 75), (10 + bar_width, 90), (0, 255, 0), -1)
                
                cv2.putText(img, f"Samples: {samples_recorded}/{target_samples} | Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Record frames continuously
                if hands_visible:
                    kpts = extract_keypoints(results)
                    frame_buffer.append(kpts)
                    
                    # Save when buffer full
                    if len(frame_buffer) >= SEQUENCE_LENGTH:
                        seq_dir = os.path.join(label_dir, str(current_seq))
                        os.makedirs(seq_dir, exist_ok=True)
                        
                        for i, kpt in enumerate(frame_buffer[:SEQUENCE_LENGTH]):
                            np.save(os.path.join(seq_dir, f"{i}.npy"), kpt)
                        
                        with open(os.path.join(seq_dir, "label.txt"), "w", encoding="utf-8") as f:
                            f.write(label)
                        
                        samples_recorded += 1
                        current_seq += 1
                        print(f"✅ Sample {samples_recorded}/{target_samples} saved!")
                        
                        if samples_recorded >= target_samples:
                            print("🎉 Target reached!")
                            break
                        
                        # Sliding window for overlap
                        frame_buffer = frame_buffer[SEQUENCE_LENGTH - gap_frames:]
            
            else:  # idle state
                bar_color = (50, 50, 50)
                cv2.rectangle(img, (0, 0), (img.shape[1], 80), bar_color, -1)
                cv2.putText(img, f"READY - Press 'R' to start", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Label: {label} | Samples: {samples_recorded}/{target_samples}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(img, f"Press R, then just keep moving until done!", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Hand indicator (always show)
            hand_status = []
            if left_hand: hand_status.append("L")
            if right_hand: hand_status.append("R")
            hand_text = f"Hands: {'+'.join(hand_status)}" if hand_status else "Hands: NONE"
            hand_color = (0, 255, 0) if hands_visible else (0, 0, 255)
            cv2.putText(img, hand_text, (img.shape[1] - 120, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
            
            cv2.imshow("Batch Collect (R=start, S=pause, Q=quit)", img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r') and state == 'idle':
                state = 'countdown'
                countdown_start = time.time()
                print(f"⏳ Countdown {delay_sec}s...")
            elif key == ord('s') and state == 'recording':
                state = 'idle'
                frame_buffer = []  # Clear buffer when paused
                print("⏸️ Paused. Press R to continue.")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if holistic is not None:
            holistic.close()
    
    print(f"\n{'='*50}")
    print(f"✅ COLLECTION COMPLETE!")
    print(f"Label: {label}")
    print(f"New samples: {samples_recorded}")
    print(f"Total samples for this label: {len(existing) + samples_recorded}")
    print(f"{'='*50}\n")

# ====== Advanced Augmentation Functions ======
def augment_time_shift(frames, max_shift=3):
    """Geser frame maju/mundur untuk variasi timing"""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(frames, shift, axis=0)

def augment_speed_variation(frames, speed_factor=None):
    """Simulasi kecepatan berbeda dengan interpolasi frame"""
    T = len(frames)
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)
    
    new_T = int(T * speed_factor)
    if new_T < 10:
        new_T = 10
    
    # Interpolasi linear
    old_indices = np.linspace(0, T - 1, new_T)
    new_frames = []
    for idx in old_indices:
        lower = int(np.floor(idx))
        upper = min(lower + 1, T - 1)
        alpha = idx - lower
        interpolated = (1 - alpha) * frames[lower] + alpha * frames[upper]
        new_frames.append(interpolated)
    
    # Pad atau trim ke SEQUENCE_LENGTH
    new_frames = np.array(new_frames)
    if len(new_frames) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(new_frames), frames.shape[1]), dtype=np.float32)
        new_frames = np.vstack([new_frames, pad])
    else:
        new_frames = new_frames[:SEQUENCE_LENGTH]
    
    return new_frames

def augment_scale(frames, scale_range=(0.85, 1.15)):
    """Perbesar/perkecil gerakan untuk variasi jarak kamera"""
    scale = np.random.uniform(*scale_range)
    # Scale hanya koordinat x,y,z (bukan visibility)
    scaled = frames.copy()
    # Pose: x,y,z di setiap 4 values
    pose = scaled[:, POSE_SLICE].reshape(-1, 33, 4)
    pose[:, :, :3] *= scale
    scaled[:, POSE_SLICE] = pose.reshape(-1, 33 * 4)
    # Hands
    scaled[:, LH_SLICE] *= scale
    scaled[:, RH_SLICE] *= scale
    return scaled

def augment_noise(frames, noise_std=0.015):
    """Tambah gaussian noise"""
    noise = np.random.normal(0, noise_std, frames.shape).astype(np.float32)
    return frames + noise

def augment_dropout(frames, drop_prob=0.05):
    """Random dropout pada beberapa frame untuk robustness"""
    mask = np.random.random(len(frames)) > drop_prob
    # Jangan drop semua, minimal 80% frame harus ada
    if mask.sum() < len(frames) * 0.8:
        return frames
    result = frames.copy()
    for i in range(len(result)):
        if not mask[i] and i > 0:
            result[i] = result[i-1]  # Copy frame sebelumnya
    return result


# ====== Loader untuk Raw Data (TANPA augmentasi) ======
def load_dataset_raw():
    """
    Load dataset TANPA augmentasi.
    Digunakan untuk split train/val sebelum augmentasi.
    Returns: X (list of arrays), labels (list of strings)
    """
    X, labels = [], []
    sentence_dirs = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    
    for sdir in sentence_dirs:
        sent = os.path.basename(sdir)
        for seq_dir in sorted(glob.glob(os.path.join(sdir, "*"))):
            frames = []
            ok = True
            for i in range(SEQUENCE_LENGTH):
                fpath = os.path.join(seq_dir, f"{i}.npy")
                if not os.path.exists(fpath):
                    ok = False
                    break
                v = np.load(fpath).astype(np.float32)
                if ZERO_FACE_ON_LOAD:
                    v = v.copy()
                    v[FACE_SLICE] = 0.0
                frames.append(v)
            if not ok:
                continue

            frames = np.stack(frames, axis=0)  # (T,1662)
            frames = normalize_batch(frames, use_face=USE_FACE)
            if ZERO_FACE_ON_LOAD or not USE_FACE:
                frames[:, FACE_SLICE] = 0.0

            # Filter: minimal 20% frame ada tangan
            lh_abs = np.abs(frames[:, LH_SLICE]).sum(axis=1)
            rh_abs = np.abs(frames[:, RH_SLICE]).sum(axis=1)
            hand_ratio = ((lh_abs + rh_abs) > 0).mean()
            if hand_ratio < 0.20:
                continue

            X.append(frames)
            labels.append(sent)
    
    if not X:
        raise RuntimeError("Dataset kosong. Jalankan mode 'collect' dulu.")
    
    print(f"[INFO] Raw dataset loaded: {len(X)} original samples (tanpa augmentasi)")
    return X, labels


def augment_samples(X_list, num_augments=4):
    """
    Augment list of samples. 
    Returns augmented samples (TIDAK termasuk original).
    """
    augmented = []
    for frames in X_list:
        # Augmentation 1: Time shift + noise (lebih variatif)
        aug1 = augment_time_shift(frames, max_shift=4)
        aug1 = augment_noise(aug1, noise_std=0.012)
        augmented.append(aug1)
        
        # Augmentation 2: Speed variation (slower/faster dynamically)
        speed_factor = np.random.uniform(0.7, 1.3)  # Lebih lebar range-nya (0.7x - 1.3x)
        aug2 = augment_speed_variation(frames, speed_factor=speed_factor)
        aug2 = augment_noise(aug2, noise_std=0.01)
        augmented.append(aug2)
        
        # Augmentation 3: Scale + slight rotation (simulate camera angle)
        aug3 = augment_scale(frames, scale_range=(0.85, 1.15))
        # Rotation belum ada func-nya, kita pakai noise lebih besar sebagai proxy
        aug3 = augment_noise(aug3, noise_std=0.015) 
        augmented.append(aug3)
        
        # Augmentation 4: Extreme dropout (simulate missing landmarks)
        aug4 = augment_scale(frames, scale_range=(0.9, 1.1))
        aug4 = augment_dropout(aug4, drop_prob=0.05) # 5% data hilang
        aug4 = augment_noise(aug4, noise_std=0.01)
        augmented.append(aug4)
    
    return augmented
def load_dataset():
    X, Y, input_lens, label_lens = [], [], [], []
    sentence_dirs = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    
    for sdir in sentence_dirs:
        sent = os.path.basename(sdir)
        for seq_dir in sorted(glob.glob(os.path.join(sdir, "*"))):
            frames = []
            ok = True
            for i in range(SEQUENCE_LENGTH):
                fpath = os.path.join(seq_dir, f"{i}.npy")
                if not os.path.exists(fpath):
                    ok = False; break
                v = np.load(fpath).astype(np.float32)
                if ZERO_FACE_ON_LOAD:
                    v = v.copy()
                    v[FACE_SLICE] = 0.0
                frames.append(v)
            if not ok:  # skip sequence tidak lengkap
                continue

            frames = np.stack(frames, axis=0)  # (T,1662)
            frames = normalize_batch(frames, use_face=USE_FACE)
            if ZERO_FACE_ON_LOAD or not USE_FACE:
                frames[:, FACE_SLICE] = 0.0

            lh_abs = np.abs(frames[:, LH_SLICE]).sum(axis=1)
            rh_abs = np.abs(frames[:, RH_SLICE]).sum(axis=1)
            hand_ratio = ((lh_abs + rh_abs) > 0).mean()
            if hand_ratio < 0.20:
                continue

            # ====== ORIGINAL DATA ======
            X.append(frames.copy())
            y_ids = text_to_ints(sent)
            Y.append(y_ids)
            input_lens.append(SEQUENCE_LENGTH)
            label_lens.append(len(y_ids))

            # ====== AUGMENTED COPIES (4x data) ======
            
            # Augmentation 1: Time shift + noise
            aug1 = augment_time_shift(frames, max_shift=3)
            aug1 = augment_noise(aug1, noise_std=0.01)
            X.append(aug1)
            Y.append(y_ids)
            input_lens.append(SEQUENCE_LENGTH)
            label_lens.append(len(y_ids))
            
            # Augmentation 2: Speed variation (slower)
            aug2 = augment_speed_variation(frames, speed_factor=0.85)
            aug2 = augment_noise(aug2, noise_std=0.008)
            X.append(aug2)
            Y.append(y_ids)
            input_lens.append(SEQUENCE_LENGTH)
            label_lens.append(len(y_ids))
            
            # Augmentation 3: Speed variation (faster) + scale
            aug3 = augment_speed_variation(frames, speed_factor=1.15)
            aug3 = augment_scale(aug3, scale_range=(0.9, 1.1))
            X.append(aug3)
            Y.append(y_ids)
            input_lens.append(SEQUENCE_LENGTH)
            label_lens.append(len(y_ids))
            
            # Augmentation 4: Scale + dropout + noise
            aug4 = augment_scale(frames, scale_range=(0.85, 1.15))
            aug4 = augment_dropout(aug4, drop_prob=0.03)
            aug4 = augment_noise(aug4, noise_std=0.012)
            X.append(aug4)
            Y.append(y_ids)
            input_lens.append(SEQUENCE_LENGTH)
            label_lens.append(len(y_ids))

    if not X:
        raise RuntimeError("Dataset kosong. Jalankan mode 'collect' dulu.")

    # Shuffle data
    indices = np.random.permutation(len(X))
    X = [X[i] for i in indices]
    Y = [Y[i] for i in indices]
    input_lens = [input_lens[i] for i in indices]
    label_lens = [label_lens[i] for i in indices]

    max_lab = max(label_lens)
    Y_pad = -np.ones((len(Y), max_lab), dtype=np.int32)
    for i, y in enumerate(Y):
        Y_pad[i, :len(y)] = y

    X = np.array(X, dtype=np.float32)
    input_lens = np.array(input_lens, dtype=np.int32).reshape(-1, 1)
    label_lens = np.array(label_lens, dtype=np.int32).reshape(-1, 1)
    
    print(f"[INFO] Dataset loaded: {len(X)} samples (dengan augmentasi 5x)")
    return X, Y_pad, input_lens, label_lens

def build_phrase_classifier(T=SEQUENCE_LENGTH, F=FEATURE_DIM, num_labels=None,
                            d_model=256, heads=8, blocks=3):
    """
    Enhanced LSTM + Transformer dengan:
    - Learnable Positional Encoding 
    - Feature Separation untuk pose/hands terpisah
    - Hand-focused attention
    """
    if num_labels is None:
        num_labels = len(LEXICON)
    
    inp = layers.Input((T, F), name="input")
    
    # ====== FEATURE SEPARATION ======
    # Pisahkan fitur: pose, left_hand, right_hand
    # Pose: 0:132, Face: 132:1536 (skip), LH: 1536:1599, RH: 1599:1662
    
    pose_feat = layers.Lambda(lambda x: x[:, :, :132], name="pose_extract")(inp)  # (T, 132)
    lh_feat = layers.Lambda(lambda x: x[:, :, 1536:1599], name="lh_extract")(inp)  # (T, 63)
    rh_feat = layers.Lambda(lambda x: x[:, :, 1599:1662], name="rh_extract")(inp)  # (T, 63)
    
    # Project each feature to same dimension
    pose_proj = layers.Dense(64, activation='relu', name="pose_proj")(pose_feat)
    lh_proj = layers.Dense(64, activation='relu', name="lh_proj")(lh_feat)
    rh_proj = layers.Dense(64, activation='relu', name="rh_proj")(rh_feat)
    
    # ====== HAND-FOCUSED PROCESSING ======
    # LSTM khusus untuk tangan (lebih penting untuk sign language)
    hands_combined = layers.Concatenate(name="hands_concat")([lh_proj, rh_proj])  # (T, 128)
    hands_lstm = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True), name="hands_lstm"
    )(hands_combined)
    hands_lstm = layers.Dropout(0.2)(hands_lstm)
    
    # LSTM untuk pose (konteks tubuh)
    pose_lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="pose_lstm"
    )(pose_proj)
    pose_lstm = layers.Dropout(0.2)(pose_lstm)
    
    # Gabungkan fitur dari hands dan pose
    combined_features = layers.Concatenate(name="feature_concat")([hands_lstm, pose_lstm])  # (T, 192+128=320)
    
    # ====== TRANSFORMER WITH LEARNABLE POSITIONAL ENCODING ======
    x = layers.Dense(d_model, activation='relu', name="trans_proj")(combined_features)
    
    # Learnable Positional Embedding (bukan sinusoidal - lebih compatible dengan Keras)
    pos_embedding = layers.Embedding(input_dim=T, output_dim=d_model, name="pos_embedding")
    positions = tf.range(start=0, limit=T, delta=1)
    pos_enc = pos_embedding(positions)  # (T, d_model)
    x = x + pos_enc  # Add position encoding
    x = layers.Dropout(0.1)(x)
    
    # Transformer blocks
    for i in range(blocks):
        # Multi-Head Self-Attention
        attn = layers.MultiHeadAttention(
            num_heads=heads, 
            key_dim=d_model//heads,
            name=f"mha_{i}"
        )(x, x)
        attn = layers.Dropout(0.1)(attn)
        x = layers.LayerNormalization(name=f"ln1_{i}")(x + attn)
        
        # Feed-Forward Network
        ff = layers.Dense(d_model * 2, activation='gelu', name=f"ff1_{i}")(x)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(0.1)(ff)
        x = layers.LayerNormalization(name=f"ln2_{i}")(x + ff)
    
    # ====== HAND ATTENTION LAYER ======
    # Attention untuk fokus ke frame dengan gerakan tangan signifikan
    hand_attn_scores = layers.Dense(1, activation='tanh', name="hand_attn_score")(x)
    hand_attn_weights = layers.Softmax(axis=1, name="hand_attn_softmax")(hand_attn_scores)
    
    # Weighted sum
    weighted = layers.Multiply(name="hand_weighted")([x, hand_attn_weights])
    hand_context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="hand_context")(weighted)
    
    # Global context
    global_pool = layers.GlobalAveragePooling1D(name="global_pool")(x)
    
    # Gabungkan hand attention + global
    final_features = layers.Concatenate(name="final_concat")([hand_context, global_pool])
    
    # ====== CLASSIFICATION HEAD ======
    final = layers.Dense(256, activation='relu', name="fc1")(final_features)
    final = layers.Dropout(0.4)(final)
    final = layers.Dense(128, activation='relu', name="fc2")(final)
    final = layers.Dropout(0.3)(final)
    out = layers.Dense(num_labels, activation='softmax', name="output")(final)
    
    model = Model(inp, out, name="SignLanguage_Enhanced")
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,  # Lower LR for stability
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_dataset_for_cls():
    X, Y_pad, _, _ = load_dataset()  # sudah normalize_batch + zero face di load_dataset()

    # --- standardize ---
    stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        mu, sigma = stats["mu"], stats["sigma"]
    else:
        mu = X.mean(axis=(0, 1), keepdims=True)
        sigma = X.std(axis=(0, 1), keepdims=True) + 1e-6
        os.makedirs(MODEL_DIR, exist_ok=True)
        np.savez(stats_path, mu=mu, sigma=sigma)

    X = (X - mu) / sigma

    # --- label: abaikan sample yang tidak ada di LEXICON ---
    y = []
    keep_idx = []
    for i, row in enumerate(Y_pad):
        txt = ints_to_text(row[row >= 0])
        if txt in LEXICON:
            y.append(LEXICON.index(txt))
            keep_idx.append(i)

    X = X[keep_idx]
    y = np.array(y, dtype=np.int64)
    return X, y


def load_dataset_for_cls_v2():
    """
    Load dataset dengan SPLIT SEBELUM AUGMENTASI.
    Ini menghindari data leakage dimana augmented samples 
    dari training bisa masuk ke validation set.
    
    Returns: X_tr, X_val, y_tr, y_val (sudah split dan augmented)
    """
    # 1. Load raw data (tanpa augmentasi)
    X_raw, labels_raw = load_dataset_raw()
    
    # 2. Filter hanya yang ada di LEXICON dan convert ke indices
    X_filtered = []
    y_filtered = []
    for x, label in zip(X_raw, labels_raw):
        if label in LEXICON:
            X_filtered.append(x)
            y_filtered.append(LEXICON.index(label))
    
    X_filtered = np.array(X_filtered, dtype=np.float32)
    y_filtered = np.array(y_filtered, dtype=np.int64)
    
    print(f"[INFO] Filtered samples in LEXICON: {len(X_filtered)}")
    
    # 3. SPLIT DULU sebelum augmentasi
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_filtered, y_filtered, 
        test_size=0.2, 
        stratify=y_filtered, 
        random_state=42
    )
    
    print(f"[INFO] Split: {len(X_tr_raw)} train, {len(X_val_raw)} val (sebelum augmentasi)")
    
    # 4. Augment HANYA training set
    X_tr_aug = augment_samples(list(X_tr_raw), num_augments=4)
    
    # Combine original + augmented untuk training
    X_tr_combined = list(X_tr_raw) + X_tr_aug
    y_tr_combined = list(y_tr) * 5  # Original + 4 augmented versions
    
    # Shuffle training data
    indices = np.random.permutation(len(X_tr_combined))
    X_tr_combined = [X_tr_combined[i] for i in indices]
    y_tr_combined = [y_tr_combined[i] for i in indices]
    
    X_tr = np.array(X_tr_combined, dtype=np.float32)
    y_tr = np.array(y_tr_combined, dtype=np.int64)
    
    # Validation tetap original (TIDAK di-augment)
    X_val = np.array(X_val_raw, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)
    
    print(f"[INFO] Final: {len(X_tr)} train (dengan augmentasi 5x), {len(X_val)} val (original only)")
    
    # 5. Standardize
    stats_path = os.path.join(MODEL_DIR, "norm_stats_v2.npz")
    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sigma = X_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(stats_path, mu=mu, sigma=sigma)
    
    X_tr = (X_tr - mu) / sigma
    X_val = (X_val - mu) / sigma
    
    return X_tr, X_val, y_tr, y_val


def train_cls():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_dataset_for_cls()

    # jumlah kelas sesuai data yang ada
    num_labels = int(np.max(y)) + 1
    model = build_phrase_classifier(num_labels=num_labels)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    ckpt = os.path.join(MODEL_DIR, "phrase_cls.h5")
    cb = [tf.keras.callbacks.ModelCheckpoint(
        ckpt, save_best_only=True, monitor="val_accuracy", mode="max"
    )]

    model.fit(
        X_tr, y_tr, epochs=100, batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=cb, verbose=2
    )
    print("Classifier tersimpan:", ckpt)

def infer_cls_realtime():
    # Try v2 stats first, fallback to original
    stats_path = os.path.join(MODEL_DIR, "norm_stats_v2.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    stats = np.load(stats_path)
    mu, sigma = stats["mu"], stats["sigma"]
    model = build_phrase_classifier(num_labels=len(LEXICON))
    model.load_weights(os.path.join(MODEL_DIR, "phrase_cls.h5"))

    cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    seq, sentence_text, stable_text = [], "", ""
    last_decoded, same_count, idle_counter = "", 0, 0
    IDLE_LIMIT = int(IDLE_FACTOR * FPS_TARGET)
    # Prediction buffer for smoothing
    prediction_hist = deque(maxlen=15)  # Average last 15 frames for stability
    hand_hist = deque(maxlen=SEQUENCE_LENGTH)
    conf = 0.0
    
    # Setup model berdasarkan mode tracking
    if USE_HYBRID_TRACKING:
        holistic = None
        print("📍 Using HYBRID tracking (Hands + Pose with temporal identity)")
    else:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("📍 Using LEGACY holistic tracking")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            img, results = mediapipe_detection(frame, holistic)
            draw_landmarks(img, results)

            kpts = extract_keypoints(results)
            seq.append(kpts);  seq = seq[-SEQUENCE_LENGTH:]
            hands_missing = (results.left_hand_landmarks is None and results.right_hand_landmarks is None)
            hand_hist.append(0 if hands_missing else 1)
            hand_ratio = sum(hand_hist) / max(1, len(hand_hist))
            idle_counter = idle_counter + 1 if hands_missing else 0

            if len(seq) == SEQUENCE_LENGTH:
                seq_raw = np.asarray(seq, dtype=np.float32)
                seq_arr = normalize_batch(seq_raw, use_face=USE_FACE)
                if not USE_FACE: seq_arr[:, FACE_SLICE] = 0.0
                seq_arr = (seq_arr - mu) / sigma
                
                # Predict
                p = model.predict(as_batch(seq_arr), verbose=0)[0]
                prediction_hist.append(p)
                
                # SMOOTHING: Average predictions over history
                p_avg = np.mean(prediction_hist, axis=0)
                pred = int(np.argmax(p_avg))
                conf = float(np.max(p_avg))
                decoded = LEXICON[pred]

                # STRICT CONFIRMATION:
                # 1. Hands must be visible > 50% in recent history
                # 2. Confidence > 0.35 (lowered to show predictions)
                if hand_ratio > 0.5 and conf > 0.35:
                    if decoded == last_decoded: 
                        same_count += 1
                    else: 
                        same_count, last_decoded = 1, decoded
                    sentence_text = decoded
                else:
                    # If uncertain or no hands, don't change sentence_text blindly
                    # But if VERY idle, clear it
                    if idle_counter > IDLE_LIMIT:
                        sentence_text = ""

            commit = (idle_counter > IDLE_LIMIT) or (same_count >= 5 and conf >= 0.7)
            if commit and sentence_text:
                stable_text = sentence_text
                speak(stable_text)
                seq, sentence_text, last_decoded, same_count = [], "", "", 0
                idle_counter, conf = 0, 0.0; hand_hist.clear(); prediction_hist.clear()

            cv2.rectangle(img, (0,0), (img.shape[1], 130), (245,117,16), -1)
            cv2.putText(img, f"CTC text: {sentence_text}", (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"FINAL: {stable_text}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # Debug: show current prediction even if not committed
            debug_pred = last_decoded if last_decoded else "(waiting...)"
            cv2.putText(img, f"conf:{conf:.2f} hands:{hand_ratio:.2f} pred:{debug_pred}", (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            cv2.imshow("Sign2Speech (q=quit)", img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if holistic is not None:
            holistic.close()


# ====== Model: Hybrid LSTM + Transformer Encoder + CTC ======
def build_ctc_model(T=SEQUENCE_LENGTH, F=FEATURE_DIM, num_classes=NUM_CLASSES, d_model=256, heads=8, blocks=3):
    inp = layers.Input(shape=(T, F), name="inputs")
    
    # ====== LSTM Branch untuk sequential patterns ======
    lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    lstm_out = layers.Dropout(0.1)(lstm_out)
    
    # ====== Transformer Branch untuk global dependencies ======
    x = layers.Dense(d_model, activation="relu")(inp)
    x = layers.Dropout(0.1)(x)
    for _ in range(blocks):
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, x)
        x = layers.LayerNormalization()(x + attn)
        ff = layers.Dense(d_model, activation="relu")(x)
        ff = layers.Dropout(0.1)(ff)
        x = layers.LayerNormalization()(x + ff)
    
    # ====== Gabungkan LSTM + Transformer ======
    combined = layers.Concatenate()([lstm_out, x])
    combined = layers.Dense(d_model, activation="relu")(combined)
    combined = layers.Dropout(0.1)(combined)
    
    y_pred = layers.Dense(num_classes, activation="softmax", name="y_pred")(combined)

    labels    = layers.Input(shape=(None,), dtype="int32", name="labels")
    input_len = layers.Input(shape=(1,),     dtype="int32", name="input_len")
    label_len = layers.Input(shape=(1,),     dtype="int32", name="label_len")

    def ctc_loss(args):
        y_true, y_pred_, in_len, lab_len = args
        return ctc_batch_cost(y_true, y_pred_, in_len, lab_len)

    loss_out = layers.Lambda(ctc_loss, name="ctc_loss")([labels, y_pred, input_len, label_len])
    train_model = Model([inp, labels, input_len, label_len], loss_out, name="train_ctc")
    train_model.add_loss(tf.reduce_mean(loss_out))
    train_model.compile(optimizer=tf.keras.optimizers.Adam(3e-4))
    infer_model = Model(inp, y_pred, name="infer_ctc")
    return train_model, infer_model

# ====== Training ======
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, Y, input_lens, label_lens = load_dataset()
    # norm stats
    mu = X.mean(axis=(0, 1), keepdims=True)
    sigma = X.std(axis=(0, 1), keepdims=True) + 1e-6
    np.savez(os.path.join(MODEL_DIR, "norm_stats.npz"), mu=mu, sigma=sigma)
    X = (X - mu) / sigma

    train_model, infer_model = build_ctc_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_WEIGHTS, save_best_only=True, monitor="val_loss", mode="min"),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")
    ]
    train_model.fit(
        x=[X, Y, input_lens, label_lens],
        y=None,
        batch_size=8,
        epochs=200,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )
    infer_model.save_weights(MODEL_WEIGHTS.replace(".h5", "_infer.h5"))
    print("Model tersimpan:", MODEL_WEIGHTS)

# ====== Levenshtein & Lexicon ======
def _levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[la][lb]

LEXICON = ["halo", "saya", "terima kasih"]  # Daftar semua frasa (harus match folder di MP_Data_Sentences)
LEXICON_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

LEXICON_PHRASES = LEXICON

use_spell_mode = False

def map_to_lexicon(text, lexicon=LEXICON, max_cer=MAX_CER_TO_MAP, margin=MARGIN_TIE):
    txt = text.strip().lower().replace(" ", "")
    if not txt: return ""
    scores = []
    for w in lexicon:
        d = _levenshtein(txt, w.replace(" ", "").lower())
        scores.append(d / max(1, len(w)))
    best_idx   = int(np.argmin(scores))
    best_score = float(scores[best_idx])
    if len(scores) > 1:
        second = float(sorted(scores)[1])
        if (second - best_score) < margin:
            return ""                 # ambigu → jangan dipaksa
    return lexicon[best_idx] if best_score <= max_cer else ""

# ====== Decoding CTC ======
def decode_probs(probs, in_len, beam_width=BEAM_WIDTH, top_paths=TOP_PATHS):
    decoded, _ = ctc_decode(
        probs,
        input_length=np.array([in_len]),
        greedy=False,
        beam_width=beam_width,
        top_paths=top_paths
    )
    cands = []
    for path in decoded:
        ids = path.numpy()[0]
        ids = [i for i in ids if 0 <= i < len(VOCAB_STR)]
        cands.append(''.join(idx2char.get(int(i), '') for i in ids).strip())

    if not cands:
        return "", ""
    # kandidat terbaik + versi yang dipetakan ke kamus
    raw_best = cands[0]
    def _score_to_lex(t):
        t0 = t.replace(" ", "").lower()
        return min(_levenshtein(t0, w.replace(" ", "").lower()) for w in LEXICON)
    lex_best = min(cands, key=_score_to_lex)
    return raw_best, map_to_lexicon(lex_best)

# ====== TTS ======
def speak(text):
    text = text.strip()
    if not text: return
    try:
        from gtts import gTTS
        import playsound
        fn = f"{uuid.uuid4().hex}.mp3"
        gTTS(text=text, lang='id').save(fn)
        playsound.playsound(fn, block=False)
        # os.remove(fn)  # opsional
        return
    except Exception as e:
        print("[TTS] gTTS gagal/offline, fallback ke pyttsx3:", e)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[TTS] pyttsx3 juga gagal:", e)

def as_batch(x: np.ndarray) -> np.ndarray:
    """x: (60,1662) -> (1,60,1662). Jika sudah (1,60,1662), dibiarkan."""
    return x if x.ndim == 3 else x[None, ...]

# ====== Mode 3: Inferensi Real-time ======
def infer_realtime():
    KEEP_TAIL = SEQUENCE_LENGTH // 2  # simpan separuh buffer setelah commit
    MIN_STABLE_FRAMES = 3  # butuh 3 frame stabil sebelum dianggap final
    # load model
    _, infer_model = build_ctc_model()
    if os.path.exists(MODEL_WEIGHTS.replace(".h5", "_infer.h5")):
        infer_model.load_weights(MODEL_WEIGHTS.replace(".h5", "_infer.h5"))
    elif os.path.exists(MODEL_WEIGHTS):
        infer_model.load_weights(MODEL_WEIGHTS)
    else:
        raise RuntimeError("Bobot model tidak ditemukan. Latih dulu dengan mode 'train'.")

    # load normalisasi
    stats = np.load(os.path.join(MODEL_DIR, "norm_stats.npz"))
    mu    = stats["mu"].reshape(-1)     # (1662,)
    sigma = stats["sigma"].reshape(-1)  # (1662,)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    seq = []
    sentence_text = ""
    stable_text   = ""
    last_decoded  = ""
    same_count    = 0
    idle_counter  = 0
    IDLE_LIMIT    = int(IDLE_FACTOR * FPS_TARGET)
    prev_conf     = 0.0
    conf          = 0.0
    raw_dec       = ""                  # <- init supaya aman dipakai di overlay

    # buffer kehadiran tangan per frame
    hand_hist = deque(maxlen=SEQUENCE_LENGTH)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img, results = mediapipe_detection(frame, holistic)
            draw_landmarks(img, results)

            # isi buffer keypoints
            kpts = extract_keypoints(results)
            seq.append(kpts)
            if len(seq) > SEQUENCE_LENGTH:
                seq = seq[-SEQUENCE_LENGTH:]

            # update info "tangan terlihat"
            hands_missing = (results.left_hand_landmarks is None and
                             results.right_hand_landmarks is None)
            hand_hist.append(0 if hands_missing else 1)
            hand_ratio = sum(hand_hist) / max(1, len(hand_hist))

            # idle counter
            idle_counter = idle_counter + 1 if hands_missing else 0

            # prediksi bila buffer penuh
            if len(seq) == SEQUENCE_LENGTH:
                seq_raw = np.asarray(seq, dtype=np.float32)  # (60,1662) RAW
                seq_arr = normalize_batch(seq_raw, use_face=USE_FACE)  # normalisasi bahu
                if not USE_FACE:
                    seq_arr[:, FACE_SLICE] = 0.0  # pastikan face nol
                # mean-std normalize
                seq_arr = (seq_arr - mu) / sigma
                probs = infer_model.predict(as_batch(seq_arr), verbose=0)

                lexicon_now = LEXICON_LETTERS if use_spell_mode else LEXICON_PHRASES
                raw_dec, lex_dec = decode_probs(probs, in_len=SEQUENCE_LENGTH)
                decoded = map_to_lexicon(lex_dec or raw_dec, lexicon=lexicon_now)

                # confidence: "seberapa tidak blank"
                blank_mean   = float(probs[0, :, BLANK_INDEX].mean())
                conf         = 1.0 - blank_mean

                # stabilizer pakai hasil lexicon
                decoded = lex_dec
                if (not decoded) and conf >= 0.70:
                    decoded = map_to_lexicon(raw_dec, max_cer=0.95, margin=1.0) or raw_dec

                if (decoded != last_decoded) and last_decoded and (same_count >= MIN_STABLE_FRAMES) \
                        and (prev_conf >= CONTENT_TH) and (hand_ratio >= HAND_VIS_TH):
                    mapped_prev = map_to_lexicon(last_decoded)
                    if mapped_prev:
                        stable_text = mapped_prev
                        speak(stable_text)
                        tail = seq[-KEEP_TAIL:] if len(seq) >= KEEP_TAIL else list(seq)
                        seq = tail
                        sentence_text = ""
                        hand_hist.clear()
                        same_count = 0

                if decoded == last_decoded:
                    same_count += 1
                else:
                    same_count  = 1
                    last_decoded = decoded
                prev_conf = conf
                sentence_text = last_decoded

            # aturan commit
            commit = (idle_counter > IDLE_LIMIT) \
                     or (same_count >= MIN_STABLE_FRAMES and conf >= CONTENT_TH and hand_ratio >= HAND_VIS_TH) \
                     or sentence_text.endswith(".") or sentence_text.endswith("!")

            if commit and sentence_text:
                mapped = map_to_lexicon(sentence_text, lexicon=(LEXICON_LETTERS if use_spell_mode else LEXICON))
                if mapped:  # hanya commit jika tidak unknown
                    stable_text = mapped
                    speak(stable_text)
                    # reset state
                    # reset state tapi simpan sebagian tail agar cepat isi ulang
                    tail = seq[-KEEP_TAIL:] if len(seq) >= KEEP_TAIL else list(seq)
                    seq = tail
                    sentence_text, last_decoded, same_count = "", "", 0
                    idle_counter, conf = 0, 0.0
                    hand_hist.clear()
                else:
                    # belum yakin, jangan commit
                    same_count = 0

            # UI
            bar_h = 110
            cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (245,117,16), -1)
            cv2.putText(img, f"CTC text: {sentence_text}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"FINAL: {stable_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, f"ctc_conf:{conf:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(img, f"raw:{raw_dec}", (250, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,255), 2)
            cv2.putText(img, f"hands_vis:{hand_ratio:.2f}", (430, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(img, f"mode:{'SPELL' if use_spell_mode else 'PHRASE'}", (640, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Sign2Speech (q=quit)", img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def debug():
    # muat data train (sudah lewat normalize_batch & zero face)
    X, Y, input_lens, label_lens = load_dataset()
    # muat norm stats
    stats = np.load(os.path.join(MODEL_DIR, "norm_stats.npz"))
    mu, sigma = stats["mu"], stats["sigma"]
    Xn = (X - mu) / sigma

    # muat model infer
    _, infer_model = build_ctc_model()
    if os.path.exists(MODEL_WEIGHTS.replace(".h5", "_infer.h5")):
        infer_model.load_weights(MODEL_WEIGHTS.replace(".h5", "_infer.h5"))
    else:
        infer_model.load_weights(MODEL_WEIGHTS)

    W = infer_model.get_layer('y_pred').get_weights()
    print("[dbg] y_pred W_norm =", np.linalg.norm(W[0]), " b_norm =", np.linalg.norm(W[1]))

    # ambil 3 contoh pertama
    for i in range(min(3, len(Xn))):
        probs = infer_model.predict(Xn[i][None, ...], verbose=0)
        raw, lex = decode_probs(probs, in_len=SEQUENCE_LENGTH)
        gt = ints_to_text(Y[i][Y[i] >= 0])
        blank_mean = float(probs[0, :, BLANK_INDEX].mean())
        print(f"[{i}] GT='{gt}' | raw='{raw}' | lex='{lex}' | content_conf={1.0-blank_mean:.2f}")

def stats():
    X, Y, _, _ = load_dataset()
    by_label = {}
    for y in Y:
        txt = ints_to_text(y[y>=0])
        by_label[txt] = by_label.get(txt, 0) + 1
    print("[stats] total seq:", len(X))
    for k,v in by_label.items():
        print(f"  {k}: {v}")

# ====== Evaluasi sederhana: CER/WER ======
def cer(ref, hyp):
    R, H = len(ref), len(hyp)
    dp = np.zeros((R+1, H+1), dtype=np.int32)
    for i in range(R+1): dp[i,0] = i
    for j in range(H+1): dp[0,j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[R,H] / max(1,R)

def wer(ref, hyp):
    r = ref.split(); h = hyp.split()
    R, H = len(r), len(h)
    dp = np.zeros((R+1, H+1), dtype=np.int32)
    for i in range(R+1): dp[i,0] = i
    for j in range(H+1): dp[0,j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[R,H] / max(1,R)

# ====== DIAGNOSTIC TOOLS ======

def train_cls_v2():
    """
    Enhanced training dengan:
    - SPLIT SEBELUM AUGMENTASI (fix data leakage!)
    - TensorBoard untuk visualisasi
    - CSV logger untuk history
    - Learning rate scheduling
    - Early stopping
    """
    import datetime
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Gunakan load_dataset_for_cls_v2 yang split SEBELUM augmentasi
    X_tr, X_val, y_tr, y_val = load_dataset_for_cls_v2()
    
    # Jumlah kelas sesuai data yang ada
    num_labels = int(np.max(y_tr)) + 1
    print(f"\n{'='*50}")
    print(f"📊 TRAINING DIAGNOSTIC INFO (FIXED DATA LEAKAGE)")
    print(f"{'='*50}")
    print(f"Num classes: {num_labels}")
    print(f"Classes: {LEXICON[:num_labels]}")
    
    # Hitung distribusi kelas di training
    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"\n📈 Training class distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = LEXICON[cls_idx] if cls_idx < len(LEXICON) else f"Class_{cls_idx}"
        print(f"  [{cls_idx}] {cls_name}: {count} samples ({100*count/len(y_tr):.1f}%)")
    
    print(f"\n🔀 Train/Val split (SEBELUM augmentasi pada training):")
    print(f"  Training: {len(X_tr)} samples (original + 4x augmented)")
    print(f"  Validation: {len(X_val)} samples (ORIGINAL only - tidak ada augmentasi!)")
    print(f"  ✅ Data leakage fixed!")
    
    model = build_phrase_classifier(num_labels=num_labels)
    
    # Callbacks lengkap
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "phrase_cls.h5"),
            save_best_only=True, 
            monitor="val_accuracy", 
            mode="max",
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1,
            write_graph=True
        ),
        # CSV logging
        tf.keras.callbacks.CSVLogger(
            os.path.join(MODEL_DIR, "training_log.csv"),
            append=False
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training...")
    print(f"📁 TensorBoard logs: {log_dir}")
    print(f"📁 CSV log: {os.path.join(MODEL_DIR, 'training_log.csv')}")
    print(f"{'='*50}\n")
    
    history = model.fit(
        X_tr, y_tr, 
        epochs=100, 
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=callbacks, 
        verbose=1  # Verbose=1 untuk melihat progress per epoch
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
    print(f"\n📊 Training history plot saved: {os.path.join(MODEL_DIR, 'training_history.png')}")
    
    # Final evaluation with comprehensive metrics
    print(f"\n{'='*50}")
    print("📋 FINAL EVALUATION")
    print(f"{'='*50}")
    train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Precision, Recall, F1-Score
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    class_names = [LEXICON[i] if i < len(LEXICON) else f"Class_{i}" for i in range(num_labels)]
    
    # Get unique labels that actually appear in y_val and y_pred
    present_labels = sorted(set(y_val) | set(y_pred))
    present_names = [class_names[i] for i in present_labels]
    
    print(f"\n📈 CLASSIFICATION REPORT (Validation Set):")
    print("-" * 50)
    print(classification_report(y_val, y_pred, labels=present_labels, target_names=present_names, digits=3, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=present_labels)
    print(f"\n🔢 CONFUSION MATRIX:")
    print("-" * 50)
    # Print header
    print("Predicted →", end="")
    for name in present_names:
        print(f" {name[:8]:>8}", end="")
    print()
    print("Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {present_names[i][:10]:10}", end="")
        for val in row:
            print(f" {val:>8}", end="")
        print()
    
    # Save confusion matrix as image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_names, yticklabels=present_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\n📊 Confusion matrix saved: {cm_path}")
    
    if train_acc > val_acc + 0.15:
        print("\n⚠️  WARNING: Possible overfitting detected!")
        print("   Train accuracy much higher than validation accuracy.")
        print("   Consider: more data, more augmentation, or regularization.")
    
    print(f"\n✅ Model saved: {os.path.join(MODEL_DIR, 'phrase_cls.h5')}")


def evaluate_model():
    """
    Evaluasi model dengan confusion matrix dan per-class metrics.
    Menampilkan kelas mana yang sering salah.
    """
    print(f"\n{'='*50}")
    print("🔍 MODEL EVALUATION - CONFUSION MATRIX")
    print(f"{'='*50}\n")
    
    # Load data
    X, y = load_dataset_for_cls()
    num_labels = int(np.max(y)) + 1
    
    # Load model
    model = build_phrase_classifier(num_labels=num_labels)
    model_path = os.path.join(MODEL_DIR, "phrase_cls.h5")
    
    if not os.path.exists(model_path):
        print("❌ Model tidak ditemukan! Latih dulu dengan train_cls atau train_cls_v2")
        return
    
    model.load_weights(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Split dengan seed yang sama
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"📊 Evaluating on {len(X_val)} validation samples...")
    
    # Predictions
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Class names
    class_names = [LEXICON[i] if i < len(LEXICON) else f"Class_{i}" for i in range(num_labels)]
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"📊 Confusion matrix saved: {cm_path}")
    
    # Classification Report
    print(f"\n{'='*50}")
    print("📋 CLASSIFICATION REPORT")
    print(f"{'='*50}")
    report = classification_report(y_val, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Per-class analysis
    print(f"\n{'='*50}")
    print("🔍 PER-CLASS ANALYSIS")
    print(f"{'='*50}")
    
    for i, cls_name in enumerate(class_names):
        mask_actual = (y_val == i)
        if mask_actual.sum() == 0:
            continue
        
        correct = (y_pred[mask_actual] == i).sum()
        total = mask_actual.sum()
        acc = correct / total
        
        # Kesalahan paling sering
        errors = y_pred[mask_actual & (y_pred != i)]
        if len(errors) > 0:
            error_counts = np.bincount(errors, minlength=num_labels)
            most_confused = np.argmax(error_counts)
            most_confused_name = class_names[most_confused]
            confused_count = error_counts[most_confused]
            print(f"[{cls_name}] Accuracy: {acc:.1%} ({correct}/{total}) | "
                  f"Most confused with: '{most_confused_name}' ({confused_count}x)")
        else:
            print(f"[{cls_name}] Accuracy: {acc:.1%} ({correct}/{total}) | Perfect! ✅")
    
    # Overall
    overall_acc = (y_pred == y_val).mean()
    print(f"\n🎯 Overall Validation Accuracy: {overall_acc:.1%}")


def review_samples(num_per_class=3):
    """
    Review sample data per kelas untuk quality check.
    Menampilkan statistik hand visibility dan konsistensi data.
    """
    print(f"\n{'='*50}")
    print("📂 DATA SAMPLE REVIEW")
    print(f"{'='*50}\n")
    
    sentence_dirs = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    
    if not sentence_dirs:
        print("❌ Dataset kosong! Rekam data dulu dengan mode 'collect'")
        return
    
    total_samples = 0
    total_valid = 0
    class_stats = {}
    
    for sdir in sentence_dirs:
        sent = os.path.basename(sdir)
        seq_dirs = sorted(glob.glob(os.path.join(sdir, "*")))
        
        valid_count = 0
        hand_ratios = []
        frame_counts = []
        
        for seq_dir in seq_dirs:
            # Cek kelengkapan frame
            frame_files = glob.glob(os.path.join(seq_dir, "*.npy"))
            frame_count = len(frame_files)
            frame_counts.append(frame_count)
            
            if frame_count < SEQUENCE_LENGTH:
                continue  # Skip incomplete
            
            # Load dan cek hand visibility
            frames = []
            for i in range(SEQUENCE_LENGTH):
                fpath = os.path.join(seq_dir, f"{i}.npy")
                if os.path.exists(fpath):
                    frames.append(np.load(fpath))
            
            if len(frames) == SEQUENCE_LENGTH:
                frames = np.stack(frames)
                lh_abs = np.abs(frames[:, LH_SLICE]).sum(axis=1)
                rh_abs = np.abs(frames[:, RH_SLICE]).sum(axis=1)
                hand_ratio = ((lh_abs + rh_abs) > 0).mean()
                hand_ratios.append(hand_ratio)
                
                if hand_ratio >= 0.20:
                    valid_count += 1
        
        class_stats[sent] = {
            'total': len(seq_dirs),
            'valid': valid_count,
            'hand_ratios': hand_ratios,
            'frame_counts': frame_counts
        }
        total_samples += len(seq_dirs)
        total_valid += valid_count
    
    # Print statistics
    print(f"📁 Dataset path: {DATA_PATH}")
    print(f"📊 Total classes: {len(sentence_dirs)}")
    print(f"📊 Total sequences: {total_samples}")
    print(f"✅ Valid sequences (hand >= 20%): {total_valid}")
    print(f"\n{'='*50}")
    print("📋 PER-CLASS STATISTICS")
    print(f"{'='*50}\n")
    
    for sent, stat in class_stats.items():
        avg_hand = np.mean(stat['hand_ratios']) if stat['hand_ratios'] else 0
        min_hand = np.min(stat['hand_ratios']) if stat['hand_ratios'] else 0
        max_hand = np.max(stat['hand_ratios']) if stat['hand_ratios'] else 0
        
        # Status indicator
        if stat['valid'] < 5:
            status = "⚠️  LOW DATA"
        elif avg_hand < 0.5:
            status = "⚠️  LOW HAND VIS"
        else:
            status = "✅ OK"
        
        print(f"[{sent}]")
        print(f"  Sequences: {stat['total']} total, {stat['valid']} valid")
        print(f"  Hand visibility: {avg_hand:.1%} avg (min: {min_hand:.1%}, max: {max_hand:.1%})")
        print(f"  Status: {status}")
        print()
    
    # Recommendations
    print(f"{'='*50}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*50}")
    
    low_data_classes = [s for s, stat in class_stats.items() if stat['valid'] < 10]
    if low_data_classes:
        print(f"\n⚠️  Classes with low data (<10 valid samples):")
        for cls in low_data_classes:
            print(f"    - '{cls}': {class_stats[cls]['valid']} valid samples")
        print("    → Consider recording more samples for these classes")
    
    imbalanced = max(s['valid'] for s in class_stats.values()) > 3 * min(s['valid'] for s in class_stats.values() if s['valid'] > 0)
    if imbalanced:
        print("\n⚠️  Class imbalance detected!")
        print("    → Consider balancing sample counts across classes")
    
    if not low_data_classes and not imbalanced:
        print("\n✅ Dataset looks good! No major issues detected.")


# ====== ONNX Runtime GPU Acceleration ======

def convert_to_onnx():
    """
    Convert Keras .h5 model ke ONNX format untuk inference cepat.
    ONNX Runtime + GPU bisa 20-50x lebih cepat dari model.predict().
    """
    import tf2onnx
    
    onnx_path = os.path.join(MODEL_DIR, "phrase_cls.onnx")
    
    print("="*50)
    print("🔄 Converting Keras model → ONNX")
    print("="*50)
    
    # Build model dengan arsitektur yang sama
    model = build_phrase_classifier(num_labels=len(LEXICON))
    h5_path = os.path.join(MODEL_DIR, "phrase_cls.h5")
    model.load_weights(h5_path)
    print(f"✅ Loaded weights from {h5_path}")
    
    # Convert ke ONNX
    input_spec = (tf.TensorSpec((1, SEQUENCE_LENGTH, FEATURE_DIM), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=input_spec,
        opset=13,
        output_path=onnx_path
    )
    
    print(f"✅ ONNX model saved to {onnx_path}")
    print(f"   Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
    print(f"\n🚀 Sekarang bisa jalankan: python main.py infer_cls_v2")


class ONNXPredictor:
    """
    Inference engine pakai ONNX Runtime.
    Otomatis pakai GPU (CUDAExecutionProvider) jika tersedia,
    fallback ke CPU jika tidak.
    
    20-50x lebih cepat dari model.predict() Keras!
    """
    def __init__(self, onnx_path):
        import onnxruntime as ort
        
        # Coba GPU dulu, fallback ke CPU
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print("🟢 ONNX Runtime: GPU (CUDA) mode")
        else:
            print("🟡 ONNX Runtime: CPU mode (CUDA not available)")
        providers.append('CPUExecutionProvider')  # Fallback
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Warmup - first call is slower
        dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        print(f"✅ ONNX model loaded and warmed up")
    
    def predict(self, x):
        """Predict dengan ONNX Runtime. x: (1, T, F) float32."""
        if x.ndim == 2:
            x = x[np.newaxis, ...]  # (T,F) -> (1,T,F)
        result = self.session.run([self.output_name], {self.input_name: x.astype(np.float32)})
        return result[0]  # (1, num_classes)


class BackgroundPredictor:
    """
    Menjalankan prediksi di background thread.
    Main thread (display) tidak pernah menunggu prediksi selesai.
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self._lock = threading.Lock()
        self._input_data = None
        self._result = None
        self._new_input = threading.Event()
        self._running = True
        self._predict_time_ms = 0.0
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self):
        while self._running:
            self._new_input.wait(timeout=0.1)
            self._new_input.clear()
            
            with self._lock:
                data = self._input_data
            
            if data is not None:
                t0 = time.perf_counter()
                result = self.predictor.predict(data)
                elapsed = (time.perf_counter() - t0) * 1000
                
                with self._lock:
                    self._result = result
                    self._predict_time_ms = elapsed
                    self._input_data = None
    
    def submit(self, x):
        """Submit data untuk prediksi (non-blocking)."""
        with self._lock:
            self._input_data = x.copy()
        self._new_input.set()
    
    def get_result(self):
        """Get latest prediction result (or None if not ready)."""
        with self._lock:
            return self._result, self._predict_time_ms
    
    def stop(self):
        self._running = False
        self._new_input.set()
        self._thread.join(timeout=2)


def infer_cls_realtime_v2():
    """
    Optimized real-time inference dengan:
    1. ONNX Runtime GPU (20-50x lebih cepat dari Keras)
    2. Background thread prediction (display selalu smooth)
    3. Frame skip (prediksi setiap N frame)
    4. MediaPipe complexity=0 (sudah di-set di HybridHandTracker)
    5. FPS counter untuk benchmarking
    """
    # Load normalization stats
    stats_path = os.path.join(MODEL_DIR, "norm_stats_v2.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    stats = np.load(stats_path)
    mu, sigma = stats["mu"], stats["sigma"]
    
    # Load ONNX model
    onnx_path = os.path.join(MODEL_DIR, "phrase_cls.onnx")
    if not os.path.exists(onnx_path):
        print("❌ ONNX model not found. Run 'python main.py convert_onnx' first.")
        print("   Falling back to Keras model...")
        return infer_cls_realtime()  # Fallback
    
    predictor = ONNXPredictor(onnx_path)
    bg_predictor = BackgroundPredictor(predictor)
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    seq = []
    sentence_text, stable_text = "", ""
    last_decoded, same_count, idle_counter = "", 0, 0
    IDLE_LIMIT = int(IDLE_FACTOR * FPS_TARGET)
    prediction_hist = deque(maxlen=15)
    hand_hist = deque(maxlen=SEQUENCE_LENGTH)
    conf = 0.0
    
    # Frame skip config
    PREDICT_EVERY = 3  # Prediksi setiap 3 frame
    frame_count = 0
    
    # FPS tracking
    fps_times = deque(maxlen=30)
    predict_ms = 0.0
    
    # Setup model berdasarkan mode tracking
    if USE_HYBRID_TRACKING:
        holistic = None
        print("📍 Using HYBRID tracking (Hands + Pose with temporal identity)")
    else:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("📍 Using LEGACY holistic tracking")
    
    print("\n" + "="*50)
    print("🚀 OPTIMIZED INFERENCE v2")
    print("="*50)
    print(f"   Engine: ONNX Runtime GPU")
    print(f"   Threading: Background prediction")
    print(f"   Frame skip: every {PREDICT_EVERY} frames")
    print(f"   MediaPipe: complexity=0 (lite)")
    print("="*50 + "\n")
    
    try:
        while True:
            t_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                break
            img, results = mediapipe_detection(frame, holistic)
            draw_landmarks(img, results)
            
            # Keypoints dikumpulkan SETIAP frame
            kpts = extract_keypoints(results)
            seq.append(kpts)
            seq = seq[-SEQUENCE_LENGTH:]
            
            hands_missing = (results.left_hand_landmarks is None and results.right_hand_landmarks is None)
            hand_hist.append(0 if hands_missing else 1)
            hand_ratio = sum(hand_hist) / max(1, len(hand_hist))
            idle_counter = idle_counter + 1 if hands_missing else 0
            
            frame_count += 1
            
            # Prediksi hanya setiap N frame (tapi keypoints tetap dikumpulkan)
            if len(seq) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY == 0:
                seq_raw = np.asarray(seq, dtype=np.float32)
                seq_arr = normalize_batch(seq_raw, use_face=USE_FACE)
                if not USE_FACE:
                    seq_arr[:, FACE_SLICE] = 0.0
                seq_arr = (seq_arr - mu) / sigma
                
                # Submit ke background thread (non-blocking!)
                bg_predictor.submit(seq_arr)
            
            # Ambil hasil prediksi terbaru (jika ada)
            result, predict_ms = bg_predictor.get_result()
            if result is not None:
                p = result[0]  # (num_classes,)
                prediction_hist.append(p)
                
                p_avg = np.mean(prediction_hist, axis=0)
                pred = int(np.argmax(p_avg))
                conf = float(np.max(p_avg))
                decoded = LEXICON[pred]
                
                if hand_ratio > 0.5 and conf > 0.35:
                    if decoded == last_decoded:
                        same_count += 1
                    else:
                        same_count, last_decoded = 1, decoded
                    sentence_text = decoded
                else:
                    if idle_counter > IDLE_LIMIT:
                        sentence_text = ""
            
            # Commit logic (lebih responsif untuk v2 dengan frame skip)
            # same_count >= 2 karena frame skip setiap 3 frame
            # conf >= 0.55 karena ONNX output lebih stabil
            commit = (idle_counter > IDLE_LIMIT) or (same_count >= 2 and conf >= 0.55)
            if commit and sentence_text:
                stable_text = sentence_text
                speak(stable_text)
                seq, sentence_text, last_decoded, same_count = [], "", "", 0
                idle_counter, conf = 0, 0.0
                hand_hist.clear()
                prediction_hist.clear()
            
            # FPS calculation
            t_end = time.perf_counter()
            fps_times.append(t_end - t_start)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
            
            # UI - header bar
            cv2.rectangle(img, (0, 0), (img.shape[1], 135), (245, 117, 16), -1)
            cv2.putText(img, f"CTC text: {sentence_text}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"FINAL: {stable_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            debug_pred = last_decoded if last_decoded else "(waiting...)"
            cv2.putText(img, f"conf:{conf:.2f} hands:{hand_ratio:.2f} pred:{debug_pred}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # FPS & performance info
            fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
            cv2.putText(img, f"FPS:{fps:.0f}", (img.shape[1] - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            cv2.putText(img, f"ONNX:{predict_ms:.0f}ms", (img.shape[1] - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img, f"v2 OPTIMIZED", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("Sign2Speech v2 (q=quit)", img)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        bg_predictor.stop()
        cap.release()
        cv2.destroyAllWindows()
        if holistic is not None:
            holistic.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("debug", help="Cek decode di data training")
    sub.add_parser("stats", help="Lihat ringkas dataset")
    sub.add_parser("train_cls", help="Latih classifier frasa")
    sub.add_parser("infer_cls", help="Inferensi real-time (classifier)")
    
    # Diagnostic tools
    sub.add_parser("train_cls_v2", help="Latih classifier dengan logging lengkap (TensorBoard, CSV)")
    sub.add_parser("evaluate", help="Evaluasi model dengan confusion matrix")
    
    # Optimized inference
    sub.add_parser("convert_onnx", help="Convert model Keras → ONNX (untuk GPU acceleration)")
    sub.add_parser("infer_cls_v2", help="Inferensi real-time OPTIMIZED (ONNX GPU + threading)")
    sub.add_parser("review", help="Review kualitas data per kelas")

    p_collect = sub.add_parser("collect", help="Rekam dataset kalimat dari webcam")
    p_collect.add_argument("--sentences", type=str, nargs="+",
                           default=["halo", "terima kasih", "apa kabar", "selamat pagi.", "saya sehat."],
                           help="Daftar kalimat target")
    p_collect.add_argument("--nseq", type=int, default=20, help="Sequence per kalimat")

    # Batch collection (Teachable Machine style)
    p_batch = sub.add_parser("collect_batch", help="Rekam data CEPAT seperti Teachable Machine (hands-free)")
    p_batch.add_argument("--label", type=str, required=True,
                         help="Label/kalimat yang akan direkam")
    p_batch.add_argument("--samples", type=int, default=50,
                         help="Jumlah sample target (default: 50)")
    p_batch.add_argument("--gap", type=int, default=5,
                         help="Gap frame antar sequence untuk sliding window (default: 5)")
    p_batch.add_argument("--delay", type=int, default=3,
                         help="Delay sebelum mulai rekam dalam detik (default: 3)")
    p_batch.add_argument("--duration", type=int, default=10,
                         help="Durasi rekam per sesi dalam detik (default: 10)")

    sub.add_parser("train", help="Latih model CTC")
    sub.add_parser("infer", help="Inferensi real-time + TTS")

    args = parser.parse_args()
    if args.mode == "collect":
        collect_dataset(args.sentences, args.nseq)
    elif args.mode == "collect_batch":
        collect_batch(args.label, target_samples=args.samples, gap_frames=args.gap,
                     delay_sec=args.delay, duration_sec=args.duration)
    elif args.mode == "train":
        train()
    elif args.mode == "infer":
        infer_realtime()
    elif args.mode == "debug":
        debug()
    elif args.mode == "stats":
        stats()
    elif args.mode == "train_cls":
        train_cls()
    elif args.mode == "infer_cls":
        infer_cls_realtime()
    elif args.mode == "train_cls_v2":
        train_cls_v2()
    elif args.mode == "evaluate":
        evaluate_model()
    elif args.mode == "review":
        review_samples()
    elif args.mode == "convert_onnx":
        convert_to_onnx()
    elif args.mode == "infer_cls_v2":
        infer_cls_realtime_v2()

if __name__ == "__main__":
    main()

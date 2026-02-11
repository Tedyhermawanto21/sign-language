"""
Sign2Speech Web Application - OPTIMIZED VERSION
Flask server untuk streaming webcam + ML inference real-time
dengan optimasi performa untuk video lebih smooth
"""

import os
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import mediapipe as mp

# Import dari main.py
from main import (
    build_phrase_classifier, extract_keypoints, normalize_batch,
    mediapipe_detection, draw_landmarks, map_to_lexicon, speak,
    LEXICON, SEQUENCE_LENGTH, FEATURE_DIM, MODEL_DIR, USE_FACE, FACE_SLICE,
    FPS_TARGET, HAND_VIS_TH, IDLE_FACTOR, FLIP_CAMERA
)
from collections import deque

app = Flask(__name__)
CORS(app)

# ====== Global State ======
current_prediction = {"text": "", "confidence": 0.0, "final": "", "mode": "PHRASE"}
is_running = False
lock = threading.Lock()

# ====== Load Model ======
print("[INFO] Loading model...")
# Coba load norm_stats_v2.npz dulu (training v2), fallback ke norm_stats.npz
stats_path = os.path.join(MODEL_DIR, "norm_stats_v2.npz")
if not os.path.exists(stats_path):
    stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
stats = np.load(stats_path)
mu, sigma = stats["mu"], stats["sigma"]
model = build_phrase_classifier(num_labels=len(LEXICON))
model.load_weights(os.path.join(MODEL_DIR, "phrase_cls.h5"))
print(f"[INFO] Model loaded successfully! Classes: {LEXICON}")

# MediaPipe - optimized settings
mp_holistic = mp.solutions.holistic

def as_batch(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 3 else x[None, ...]


# ====== Background Prediction Thread ======
prediction_data = {
    "seq": [],
    "ready": False,
    "result": None,
    "conf": 0.0
}
pred_lock = threading.Lock()


def prediction_worker():
    """Background thread untuk prediksi - tidak blocking video"""
    global prediction_data
    
    while is_running:
        with pred_lock:
            if prediction_data["ready"] and len(prediction_data["seq"]) == SEQUENCE_LENGTH:
                seq_raw = np.asarray(prediction_data["seq"], dtype=np.float32)
                seq_arr = normalize_batch(seq_raw, use_face=USE_FACE)
                if not USE_FACE:
                    seq_arr[:, FACE_SLICE] = 0.0
                seq_arr = (seq_arr - mu) / sigma
                
                p = model.predict(as_batch(seq_arr), verbose=0)[0]
                pred = int(np.argmax(p))
                conf = float(np.max(p))
                
                prediction_data["result"] = LEXICON[pred]
                prediction_data["conf"] = conf
                prediction_data["ready"] = False
        
        time.sleep(0.05)  # 50ms interval - reduce CPU usage


def generate_frames():
    """Generator untuk streaming video dengan prediksi overlay - OPTIMIZED"""
    global current_prediction, is_running, prediction_data
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for Windows - faster
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer = less latency
    
    seq = []
    stable_text = ""
    last_decoded = ""
    same_count = 0
    idle_counter = 0
    IDLE_LIMIT = int(IDLE_FACTOR * FPS_TARGET)
    hand_hist = deque(maxlen=SEQUENCE_LENGTH)
    conf = 0.0
    sentence_text = ""
    locked_prediction = ""  # Prediksi yang di-lock saat confidence tinggi
    locked_conf = 0.0
    
    frame_count = 0
    PREDICT_EVERY = 2  # Prediksi setiap 2 frame untuk smooth video
    
    is_running = True
    
    # Start prediction worker thread
    pred_thread = threading.Thread(target=prediction_worker, daemon=True)
    pred_thread.start()
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=lite, 1=full, 2=heavy
    ) as holistic:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # mediapipe_detection sudah handle flip jika FLIP_CAMERA=True
            img, results = mediapipe_detection(frame, holistic)
            draw_landmarks(img, results)
            
            # Extract keypoints setiap frame untuk buffer
            kpts = extract_keypoints(results)
            seq.append(kpts)
            seq = seq[-SEQUENCE_LENGTH:]
            
            # Track hand visibility
            hands_missing = (results.left_hand_landmarks is None and 
                           results.right_hand_landmarks is None)
            hand_hist.append(0 if hands_missing else 1)
            hand_ratio = sum(hand_hist) / max(1, len(hand_hist))
            idle_counter = idle_counter + 1 if hands_missing else 0
            
            # Kirim ke prediction thread setiap N frame
            if len(seq) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY == 0:
                with pred_lock:
                    prediction_data["seq"] = list(seq)
                    prediction_data["ready"] = True
            
            # Ambil hasil prediksi dari thread
            with pred_lock:
                if prediction_data["result"]:
                    decoded = prediction_data["result"]
                    conf = prediction_data["conf"]
                    
                    if decoded == last_decoded:
                        same_count += 1
                    else:
                        same_count = 1
                        last_decoded = decoded
                    
                    sentence_text = decoded
                    
                    # Update global prediction
                    with lock:
                        current_prediction["text"] = sentence_text
                        current_prediction["confidence"] = conf
                    
                    # LOCK prediction ketika confidence cukup dan stabil
                    # Ini akan di-commit nanti ketika tangan hilang
                    if same_count >= 3 and conf >= 0.50 and hand_ratio >= HAND_VIS_TH:
                        locked_prediction = sentence_text
                        locked_conf = conf
            
            # Commit logic - gunakan LOCKED prediction, bukan current
            # Commit ketika tangan hilang (idle) DAN ada locked prediction
            if idle_counter > IDLE_LIMIT and locked_prediction:
                mapped = map_to_lexicon(locked_prediction, lexicon=LEXICON)
                if mapped:
                    stable_text = mapped
                    with lock:
                        current_prediction["final"] = stable_text
                    # Speak in background thread
                    threading.Thread(target=speak, args=(stable_text,), daemon=True).start()
                    # Reset
                    seq = []
                    sentence_text = ""
                    last_decoded = ""
                    same_count = 0
                    idle_counter = 0
                    conf = 0.0
                    locked_prediction = ""
                    locked_conf = 0.0
                    hand_hist.clear()
                    with pred_lock:
                        prediction_data["result"] = None
            
            # Draw overlay - simplified for performance
            cv2.rectangle(img, (0, 0), (640, 70), (20, 20, 30), -1)
            
            # Text overlays
            cv2.putText(img, f"Prediksi: {sentence_text}", (15, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Conf: {conf:.0%} | Hands: {hand_ratio:.0%}", (15, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 200), 1)
            
            if stable_text:
                cv2.putText(img, f"FINAL: {stable_text}", (350, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 2)
            
            # Encode frame dengan quality lebih rendah untuk speed
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    is_running = False


@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', lexicon=LEXICON)


@app.route('/video_feed')
def video_feed():
    """Streaming video endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def get_prediction():
    """API endpoint untuk mendapatkan prediksi terkini"""
    with lock:
        return jsonify(current_prediction)


@app.route('/speak', methods=['POST'])
def speak_text():
    """Endpoint untuk TTS manual"""
    data = request.get_json()
    text = data.get('text', '')
    if text:
        threading.Thread(target=speak, args=(text,), daemon=True).start()
        return jsonify({"status": "ok", "text": text})
    return jsonify({"status": "error", "message": "No text provided"}), 400


@app.route('/stop', methods=['POST'])
def stop_stream():
    """Stop video stream"""
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🤟 Sign2Speech Web Application (OPTIMIZED)")
    print("="*50)
    print("Buka browser di: http://localhost:5000")
    print("Tekan Ctrl+C untuk berhenti")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

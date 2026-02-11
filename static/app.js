/**
 * Sign2Speech - Frontend JavaScript
 * Real-time prediction updates dan interaktivitas
 */

// ====== State ======
let lastFinal = '';
let history = [];

// ====== DOM Elements ======
const currentPrediction = document.getElementById('currentPrediction');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const finalResult = document.getElementById('finalResult');
const speakBtn = document.getElementById('speakBtn');
const historyList = document.getElementById('historyList');
const clearHistory = document.getElementById('clearHistory');
const lexiconItems = document.querySelectorAll('.lexicon-item');

// ====== Polling untuk Prediksi ======
async function fetchPrediction() {
    try {
        const response = await fetch('/prediction');
        const data = await response.json();

        // Update current prediction
        if (data.text) {
            currentPrediction.textContent = data.text;
            currentPrediction.classList.add('active');

            // Highlight matching lexicon
            highlightLexicon(data.text);
        } else {
            currentPrediction.textContent = 'Menunggu gerakan...';
            currentPrediction.classList.remove('active');
            clearLexiconHighlight();
        }

        // Update confidence bar
        const confidence = Math.round(data.confidence * 100);
        confidenceFill.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence}%`;

        // Update color based on confidence
        if (confidence >= 70) {
            confidenceFill.style.background = 'linear-gradient(90deg, #22c55e, #06b6d4)';
        } else if (confidence >= 40) {
            confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #eab308)';
        } else {
            confidenceFill.style.background = 'linear-gradient(90deg, #6366f1, #8b5cf6)';
        }

        // Check for new final result
        if (data.final && data.final !== lastFinal) {
            lastFinal = data.final;
            updateFinalResult(data.final);
            addToHistory(data.final);
        }

    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

// ====== Update Final Result ======
function updateFinalResult(text) {
    finalResult.textContent = text;
    finalResult.classList.add('pulse');

    // Add celebration effect
    createConfetti();

    setTimeout(() => {
        finalResult.classList.remove('pulse');
    }, 500);
}

// ====== History ======
function addToHistory(text) {
    const now = new Date();
    const time = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });

    history.unshift({ text, time });

    // Limit history
    if (history.length > 20) {
        history.pop();
    }

    renderHistory();
}

function renderHistory() {
    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-history">Belum ada riwayat</p>';
        return;
    }

    historyList.innerHTML = history.map(item => `
        <div class="history-item">
            <span class="time">${item.time}</span>
            <span class="text">${item.text}</span>
        </div>
    `).join('');
}

// ====== Lexicon Highlight ======
function highlightLexicon(text) {
    lexiconItems.forEach(item => {
        if (item.dataset.phrase.toLowerCase() === text.toLowerCase()) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

function clearLexiconHighlight() {
    lexiconItems.forEach(item => item.classList.remove('active'));
}

// ====== TTS ======
async function speakText(text) {
    if (!text || text === '-') return;

    try {
        await fetch('/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        // Visual feedback
        speakBtn.style.transform = 'scale(0.9)';
        setTimeout(() => {
            speakBtn.style.transform = 'scale(1)';
        }, 150);

    } catch (error) {
        console.error('TTS error:', error);
    }
}

// ====== Confetti Effect ======
function createConfetti() {
    const colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b'];

    for (let i = 0; i < 30; i++) {
        const confetti = document.createElement('div');
        confetti.style.cssText = `
            position: fixed;
            width: 10px;
            height: 10px;
            background: ${colors[Math.floor(Math.random() * colors.length)]};
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            left: ${50 + (Math.random() - 0.5) * 30}%;
            top: 50%;
            animation: confettiFall 1s ease-out forwards;
        `;
        document.body.appendChild(confetti);

        setTimeout(() => confetti.remove(), 1000);
    }
}

// Add confetti animation
const style = document.createElement('style');
style.textContent = `
    @keyframes confettiFall {
        0% {
            opacity: 1;
            transform: translateY(0) rotate(0deg) scale(1);
        }
        100% {
            opacity: 0;
            transform: translateY(${Math.random() * 200 + 100}px) rotate(${Math.random() * 720}deg) scale(0);
        }
    }
    
    .pulse {
        animation: resultPulse 0.5s ease-out;
    }
    
    @keyframes resultPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(style);

// ====== Event Listeners ======
speakBtn.addEventListener('click', () => {
    speakText(finalResult.textContent);
});

clearHistory.addEventListener('click', () => {
    history = [];
    renderHistory();
});

lexiconItems.forEach(item => {
    item.addEventListener('click', () => {
        speakText(item.dataset.phrase);
    });
});

// ====== Start Polling ======
setInterval(fetchPrediction, 200);

// Initial fetch
fetchPrediction();

console.log('🤟 Sign2Speech loaded successfully!');

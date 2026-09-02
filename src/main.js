import './style.css';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as faceapi from '@vladmandic/face-api';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- STABLE STATE & GLOBALS ---
let stream = null;
let cocoModel = null;
let handDetector = null;
let inferenceRequestId = null;

let isPaused = false;
let modelsLoaded = false;
let currentDeviceId = '';

// UI settings
let detectObjectsEnabled = true;
let detectFacesEnabled = true;
let detectHandsEnabled = true;
let detectGesturesEnabled = true;
let spiderEffectEnabled = true;
let showJointNumbersEnabled = true;
let detectDrowsinessEnabled = true;
let audioAlarmEnabled = true;
let sfxEnabled = true;
let voiceAssistantEnabled = true;
let airCanvasEnabled = true;
let airBrushColor = '#00f2fe';
let airBrushSize = 6;
let lastAirPoint = null;
let hudActive = true;
let confidenceThreshold = 0.40; // default 40%

// SFX & Voice Timestamps
let lastSpeechTime = 0;
let lastWebShooterSFXTime = 0;
let lastChimeSFXTime = 0;
let lastLockSFXTime = 0;

// Attention & Drowsiness Tracker State
let closedEyeStartTime = 0;
let isEyeClosedState = false;
let isDrowsyAlarmActive = false;
let sessionBlinkCount = 0;
let currentEarValue = 0.30;
let currentAttentionIndex = 100;
let audioCtx = null;
let lastAudioBeepTime = 0;

// Active tab and DOM update throttling
let activeTab = 'tab-settings';
let lastDomUpdateTime = 0;

// Performance stats
let frameCount = 0;
let fps = 0;
let fpsTimer = Date.now();

// --- MULTI-FRAME SMOOTHING ---
const DETECTION_HISTORY_FRAMES = 3;
const detectionHistory = [];

// --- PRE-PROCESSING CANVAS ---
const preprocessCanvas = document.createElement('canvas');
const preprocessCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });

// Biometrics & Session Statistics
const sessionStats = {
    uniqueFacesCount: 0,
    trackedAges: {},
    emotionHits: {
        happy: 0,
        neutral: 0,
        surprised: 0,
        sad: 0,
        angry: 0
    },
    emotionTotal: 0
};

// --- DOM ELEMENTS ---
const elements = {
    video: document.getElementById('webcam'),
    canvas: document.getElementById('overlay'),
    drawingCanvas: document.getElementById('drawing-canvas'),
    airToolbar: document.getElementById('air-canvas-toolbar'),
    btnToggleAir: document.getElementById('btn-toggle-air'),
    colorSwatches: document.querySelectorAll('.color-swatch'),
    brushSizeSlider: document.getElementById('brush-size-slider'),
    btnClearCanvas: document.getElementById('btn-clear-canvas'),

    // Master Feature Hub Modal
    btnOpenHub: document.getElementById('btn-open-hub'),
    btnCloseHub: document.getElementById('btn-close-hub'),
    featureHubModal: document.getElementById('feature-hub-modal'),

    fpsCounter: document.getElementById('fps-counter'),
    backendType: document.getElementById('backend-type'),
    systemStatus: document.getElementById('system-status'),
    resolution: document.getElementById('viewport-resolution'),
    modelLoader: document.getElementById('model-loader'),
    errorBanner: document.getElementById('error-banner'),
    errorMessage: document.getElementById('error-message'),
    btnRetry: document.getElementById('btn-retry'),

    // HUD Buttons
    btnPause: document.getElementById('btn-pause-stream'),
    btnScreenshot: document.getElementById('btn-screenshot'),
    btnToggleHud: document.getElementById('btn-toggle-hud'),
    viewportCard: document.querySelector('.viewport-card'),

    // Settings Config
    cameraSelect: document.getElementById('camera-select'),
    toggleObjects: document.getElementById('toggle-objects'),
    toggleFaces: document.getElementById('toggle-faces'),
    toggleHands: document.getElementById('toggle-hands'),
    toggleGestures: document.getElementById('toggle-gestures'),
    toggleSpider: document.getElementById('toggle-spider'),
    toggleNumbers: document.getElementById('toggle-numbers'),
    toggleDrowsiness: document.getElementById('toggle-drowsiness'),
    toggleAudioAlarm: document.getElementById('toggle-audio-alarm'),
    toggleSfx: document.getElementById('toggle-sfx'),
    toggleVoice: document.getElementById('toggle-voice'),
    confidenceSlider: document.getElementById('confidence-threshold'),
    confidenceVal: document.getElementById('confidence-val'),

    // Sidebar Logs & Badges
    totalObjectsBadge: document.getElementById('total-objects-badge'),
    faceCountBadge: document.getElementById('face-count-badge'),
    detectionsContainer: document.getElementById('detections-container'),
    faceAnalyticsContainer: document.getElementById('face-analytics-container'),

    // Tab nav
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    tabObjCount: document.getElementById('tab-obj-count'),
    tabFaceCount: document.getElementById('tab-face-count'),

    // Telemetry Dashboard
    telUniqueFaces: document.getElementById('tel-unique-faces'),
    telAvgAge: document.getElementById('tel-avg-age'),
    telBlinkCount: document.getElementById('tel-blink-count'),
    telDrowsyStatus: document.getElementById('tel-drowsy-status'),
    barEar: document.getElementById('bar-ear'),
    telEarVal: document.getElementById('tel-ear-val'),
    barAttention: document.getElementById('bar-attention'),
    telAttentionVal: document.getElementById('tel-attention-val'),
    drowsinessAlertOverlay: document.getElementById('drowsiness-alert-overlay'),
    toastContainer: document.getElementById('toast-container'),

    // Theme Toggle Elements
    themeToggle: document.getElementById('theme-toggle'),
    sunIcon: document.getElementById('theme-icon-sun'),
    moonIcon: document.getElementById('theme-icon-moon'),

    // Emotion Bars
    barHappy: document.getElementById('bar-happy'),
    pctHappy: document.getElementById('pct-happy'),
    barNeutral: document.getElementById('bar-neutral'),
    pctNeutral: document.getElementById('pct-neutral'),
    barSurprised: document.getElementById('bar-surprised'),
    pctSurprised: document.getElementById('pct-surprised'),
    barSad: document.getElementById('bar-sad'),
    pctSad: document.getElementById('pct-sad'),
    barAngry: document.getElementById('bar-angry'),
    pctAngry: document.getElementById('pct-angry')
};

// --- TAB SWITCHING ---
function initTabs() {
    elements.tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.tab;
            activeTab = targetId;

            elements.tabButtons.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            elements.tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            const pane = document.getElementById(targetId);
            if (pane) pane.classList.add('active');

            lastDomUpdateTime = 0;
        });
    });
}

// --- THEME MANAGEMENT ---
function initTheme() {
    const savedTheme = localStorage.getItem('theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    applyTheme(savedTheme);

    elements.themeToggle.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme') || 'light';
        applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    if (theme === 'dark') {
        elements.sunIcon.classList.remove('hidden');
        elements.moonIcon.classList.add('hidden');
    } else {
        elements.sunIcon.classList.add('hidden');
        elements.moonIcon.classList.remove('hidden');
    }
}

function getThemeColor(variableName, fallback) {
    return getComputedStyle(document.body).getPropertyValue(variableName).trim() || fallback;
}

// --- INTERSECTION OVER UNION (IoU) TRACKER ---
class FaceTracker {
    constructor() {
        this.trackedFaces = [];
        this.nextId = 1;
    }

    update(currentDetections) {
        const now = Date.now();
        const matchedTrackIds = new Set();
        const activeTracks = [];

        for (const det of currentDetections) {
            const currentBox = det.detection.box;
            let bestMatchIndex = -1;
            let maxIoU = 0;

            for (let i = 0; i < this.trackedFaces.length; i++) {
                const track = this.trackedFaces[i];
                if (matchedTrackIds.has(track.id)) continue;

                const iou = this.getIoU(currentBox, track.box);
                if (iou > maxIoU && iou > 0.35) {
                    maxIoU = iou;
                    bestMatchIndex = i;
                }
            }

            if (bestMatchIndex !== -1) {
                const match = this.trackedFaces[bestMatchIndex];
                match.box = currentBox;

                if (!sessionStats.trackedAges[match.id]) {
                    sessionStats.trackedAges[match.id] = [];
                }
                sessionStats.trackedAges[match.id].push(det.age);
                if (sessionStats.trackedAges[match.id].length > 15) {
                    sessionStats.trackedAges[match.id].shift();
                }
                const ageSum = sessionStats.trackedAges[match.id].reduce((a, b) => a + b, 0);
                match.age = ageSum / sessionStats.trackedAges[match.id].length;

                match.gender = det.gender;
                match.expressions = det.expressions;
                match.lastSeen = now;

                matchedTrackIds.add(match.id);
                activeTracks.push(match);
            } else {
                const newId = this.nextId++;
                sessionStats.trackedAges[newId] = [det.age];
                sessionStats.uniqueFacesCount++;

                const newTrack = {
                    id: newId,
                    box: currentBox,
                    age: det.age,
                    gender: det.gender,
                    expressions: det.expressions,
                    firstSeen: now,
                    lastSeen: now
                };

                activeTracks.push(newTrack);
                playTargetLockSFX();
                speakVoiceAlert(`Subject #${newId} identified`);
                showToast(`Person #${newId} detected (~${Math.round(det.age)}y/o, ${det.gender})`, 'success');
            }
        }

        const lostTracks = this.trackedFaces.filter(
            track => !matchedTrackIds.has(track.id) && (now - track.lastSeen < 1500)
        );

        this.trackedFaces = activeTracks.concat(lostTracks);
        return activeTracks;
    }

    getIoU(box1, box2) {
        const xA = Math.max(box1.x, box2.x);
        const yA = Math.max(box1.y, box2.y);
        const xB = Math.min(box1.x + box1.width, box2.x + box2.width);
        const yB = Math.min(box1.y + box1.height, box2.y + box2.height);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        if (interArea === 0) return 0;

        const box1Area = box1.width * box1.height;
        const box2Area = box2.width * box2.height;

        return interArea / (box1Area + box2Area - interArea);
    }
}

const faceTracker = new FaceTracker();

class ObjectTracker {
    constructor() {
        this.trackedObjects = [];
        this.nextId = 1;
    }

    update(currentDetections) {
        const now = Date.now();
        const matchedTrackIds = new Set();
        const activeTracks = [];

        for (const det of currentDetections) {
            const currentBox = det.bbox;
            let bestMatchIndex = -1;
            let maxIoU = 0;

            for (let i = 0; i < this.trackedObjects.length; i++) {
                const track = this.trackedObjects[i];
                if (matchedTrackIds.has(track.id)) continue;
                if (track.class !== det.class) continue;

                const iou = this.getIoU(currentBox, track.bbox);
                if (iou > maxIoU && iou > 0.25) {
                    maxIoU = iou;
                    bestMatchIndex = i;
                }
            }

            if (bestMatchIndex !== -1) {
                const match = this.trackedObjects[bestMatchIndex];
                const alpha = 0.5;
                match.bbox = [
                    alpha * currentBox[0] + (1 - alpha) * match.bbox[0],
                    alpha * currentBox[1] + (1 - alpha) * match.bbox[1],
                    alpha * currentBox[2] + (1 - alpha) * match.bbox[2],
                    alpha * currentBox[3] + (1 - alpha) * match.bbox[3]
                ];
                match.score = det.score;
                match.lastSeen = now;
                matchedTrackIds.add(match.id);
                activeTracks.push(match);
            } else {
                const newId = this.nextId++;
                const newTrack = {
                    id: newId,
                    class: det.class,
                    bbox: currentBox,
                    score: det.score,
                    firstSeen: now,
                    lastSeen: now
                };
                activeTracks.push(newTrack);
            }
        }

        const lostTracks = this.trackedObjects.filter(
            track => !matchedTrackIds.has(track.id) && (now - track.lastSeen < 600)
        );

        this.trackedObjects = activeTracks.concat(lostTracks);
        return this.trackedObjects;
    }

    getIoU(box1, box2) {
        const xA = Math.max(box1[0], box2[0]);
        const yA = Math.max(box1[1], box2[1]);
        const xB = Math.min(box1[0] + box1[2], box2[0] + box2[2]);
        const yB = Math.min(box1[1] + box1[3], box2[1] + box2[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        if (interArea === 0) return 0;

        const box1Area = box1[2] * box1[3];
        const box2Area = box2[2] * box2[3];

        return interArea / (box1Area + box2Area - interArea);
    }
}

const objectTracker = new ObjectTracker();

// --- TOAST NOTIFICATIONS ---
function showToast(message, type = 'default') {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'success' ? 'toast-success' : type === 'error' ? 'toast-error' : ''}`;

    const icon = type === 'success' ? '🟢' : type === 'error' ? '🔴' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span> <p>${message}</p>`;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// --- ATTENTION & DROWSINESS ALGORITHM ENGINE ---
function calculateEuclideanDistance(pt1, pt2) {
    return Math.hypot(pt1.x - pt2.x, pt1.y - pt2.y);
}

function calculateEAR(landmarks) {
    if (!landmarks || !landmarks.positions || landmarks.positions.length < 68) {
        return 0.30;
    }
    const pts = landmarks.positions;

    // Left eye: indices 36 to 41
    const l1 = calculateEuclideanDistance(pts[37], pts[41]);
    const l2 = calculateEuclideanDistance(pts[38], pts[40]);
    const l3 = calculateEuclideanDistance(pts[36], pts[39]);
    const leftEar = (l1 + l2) / (2.0 * (l3 || 1));

    // Right eye: indices 42 to 47
    const r1 = calculateEuclideanDistance(pts[43], pts[47]);
    const r2 = calculateEuclideanDistance(pts[44], pts[46]);
    const r3 = calculateEuclideanDistance(pts[42], pts[45]);
    const rightEar = (r1 + r2) / (2.0 * (r3 || 1));

    return (leftEar + rightEar) / 2.0;
}

function calculateAttentionScore(landmarks) {
    if (!landmarks || !landmarks.positions || landmarks.positions.length < 68) {
        return 100;
    }
    const pts = landmarks.positions;
    const eyeSpan = calculateEuclideanDistance(pts[36], pts[45]);
    if (eyeSpan === 0) return 100;

    const eyeMidX = (pts[36].x + pts[45].x) / 2;
    const noseX = pts[30].x;

    const offsetRatio = Math.abs(noseX - eyeMidX) / eyeSpan;
    const score = Math.max(0, Math.min(100, Math.round((1 - (offsetRatio / 0.35)) * 100)));
    return score;
}

function getAudioCtx() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function speakVoiceAlert(text) {
    if (!voiceAssistantEnabled || !('speechSynthesis' in window)) return;
    const now = Date.now();
    if (now - lastSpeechTime < 2500) return;
    lastSpeechTime = now;

    try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Female') || v.name.includes('Samantha')));
        if (preferredVoice) utterance.voice = preferredVoice;

        window.speechSynthesis.speak(utterance);
    } catch (err) {
        console.warn('Voice synthesis exception:', err);
    }
}

function playWebShooterSFX() {
    if (!sfxEnabled) return;
    const now = Date.now();
    if (now - lastWebShooterSFXTime < 450) return;
    lastWebShooterSFXTime = now;

    try {
        const ctx = getAudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(960, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(180, ctx.currentTime + 0.3);

        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.3);
    } catch (err) {
        console.warn('SFX error:', err);
    }
}

function playGestureChimeSFX() {
    if (!sfxEnabled) return;
    const now = Date.now();
    if (now - lastChimeSFXTime < 600) return;
    lastChimeSFXTime = now;

    try {
        const ctx = getAudioCtx();
        const notes = [523.25, 659.25, 783.99];
        notes.forEach((freq, idx) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            const startTime = ctx.currentTime + idx * 0.08;

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, startTime);

            gain.gain.setValueAtTime(0.18, startTime);
            gain.gain.exponentialRampToValueAtTime(0.01, startTime + 0.22);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(startTime);
            osc.stop(startTime + 0.22);
        });
    } catch (err) {
        console.warn('Chime SFX error:', err);
    }
}

function playPalmWipeSFX() {
    if (!sfxEnabled) return;
    try {
        const ctx = getAudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(450, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(120, ctx.currentTime + 0.35);

        gain.gain.setValueAtTime(0.28, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.35);
    } catch (err) {
        console.warn('Wipe SFX error:', err);
    }
}

function playTargetLockSFX() {
    if (!sfxEnabled) return;
    const now = Date.now();
    if (now - lastLockSFXTime < 1200) return;
    lastLockSFXTime = now;

    try {
        const ctx = getAudioCtx();
        [587.33, 880].forEach((freq, idx) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            const startTime = ctx.currentTime + idx * 0.09;

            osc.type = 'square';
            osc.frequency.setValueAtTime(freq, startTime);

            gain.gain.setValueAtTime(0.1, startTime);
            gain.gain.exponentialRampToValueAtTime(0.01, startTime + 0.12);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(startTime);
            osc.stop(startTime + 0.12);
        });
    } catch (err) {
        console.warn('Target lock SFX error:', err);
    }
}

function triggerSynthAlarmSound() {
    if (!audioAlarmEnabled) return;
    const now = Date.now();
    if (now - lastAudioBeepTime < 320) return;
    lastAudioBeepTime = now;

    try {
        const ctx = getAudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.22);

        gain.gain.setValueAtTime(0.35, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.22);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.22);
    } catch (err) {
        console.warn('Audio alarm exception:', err);
    }
}

function processDrowsinessAndAttention(faceDetections) {
    if (!detectDrowsinessEnabled || !faceDetections || faceDetections.length === 0) {
        isDrowsyAlarmActive = false;
        if (elements.drowsinessAlertOverlay) {
            elements.drowsinessAlertOverlay.classList.add('hidden');
        }
        return;
    }

    const primaryFace = faceDetections[0];
    if (!primaryFace || !primaryFace.landmarks) return;

    const ear = calculateEAR(primaryFace.landmarks);
    const attention = calculateAttentionScore(primaryFace.landmarks);
    currentEarValue = ear;
    currentAttentionIndex = attention;

    const now = Date.now();
    const DROWSY_EAR_THRESHOLD = 0.20;
    const OPEN_EAR_THRESHOLD = 0.23;

    if (ear < DROWSY_EAR_THRESHOLD) {
        if (!isEyeClosedState) {
            isEyeClosedState = true;
            closedEyeStartTime = now;
        } else {
            const closedDuration = now - closedEyeStartTime;
            if (closedDuration >= 1300) {
                isDrowsyAlarmActive = true;
                triggerSynthAlarmSound();
            }
        }
    } else if (ear >= OPEN_EAR_THRESHOLD) {
        if (isEyeClosedState) {
            const duration = now - closedEyeStartTime;
            if (duration > 80 && duration < 800) {
                sessionBlinkCount++;
            }
            isEyeClosedState = false;
        }
        isDrowsyAlarmActive = false;
    }

    if (elements.drowsinessAlertOverlay) {
        if (isDrowsyAlarmActive) {
            elements.drowsinessAlertOverlay.classList.remove('hidden');
        } else {
            elements.drowsinessAlertOverlay.classList.add('hidden');
        }
    }
}

// --- AIR CANVAS DRAWING ENGINE ---
let lastPalmClearTime = 0;

function clearAirCanvas() {
    if (!elements.drawingCanvas) return;
    const drawCtx = elements.drawingCanvas.getContext('2d');
    drawCtx.clearRect(0, 0, elements.drawingCanvas.width, elements.drawingCanvas.height);
    playPalmWipeSFX();
    speakVoiceAlert('Air canvas wiped clean');
    showToast('Air Canvas wiped clean 🖐️!', 'success');
}

function processAirCanvas(mappedHands) {
    if (!airCanvasEnabled || !elements.drawingCanvas) return;
    const drawCtx = elements.drawingCanvas.getContext('2d');

    if (!mappedHands || mappedHands.length === 0) {
        lastAirPoint = null;
        return;
    }

    let isPointing = false;
    const now = Date.now();

    mappedHands.forEach(hand => {
        const keypoints = hand.keypoints;
        if (!keypoints || keypoints.length < 21) return;

        const gestureObj = classifyGesture(keypoints);
        const gestureName = gestureObj && gestureObj.name ? gestureObj.name.toUpperCase() : '';

        // Open Palm gesture detection (all 4 fingers raised)
        const allFingersUp = keypoints[8].y < keypoints[6].y &&
                             keypoints[12].y < keypoints[10].y &&
                             keypoints[16].y < keypoints[14].y &&
                             keypoints[20].y < keypoints[18].y;

        const isOpenPalm = gestureName.includes('PALM') || allFingersUp;

        if (isOpenPalm) {
            lastAirPoint = null;
            if (now - lastPalmClearTime > 1200) {
                lastPalmClearTime = now;
                clearAirCanvas();
            }
            return;
        }

        // Check if index finger is extended while middle finger is curled
        const indexExtended = keypoints[8].y < keypoints[6].y && keypoints[12].y > keypoints[10].y;

        if (gestureName.includes('POINTING') || indexExtended) {
            isPointing = true;
            const indexTip = keypoints[8];

            drawCtx.save();
            drawCtx.strokeStyle = airBrushColor;
            drawCtx.fillStyle = airBrushColor;
            drawCtx.lineWidth = airBrushSize;
            drawCtx.lineCap = 'round';
            drawCtx.lineJoin = 'round';
            drawCtx.shadowColor = airBrushColor;
            drawCtx.shadowBlur = airBrushSize * 1.5;

            if (lastAirPoint) {
                drawCtx.beginPath();
                drawCtx.moveTo(lastAirPoint.x, lastAirPoint.y);
                drawCtx.lineTo(indexTip.x, indexTip.y);
                drawCtx.stroke();
            }

            drawCtx.beginPath();
            drawCtx.arc(indexTip.x, indexTip.y, airBrushSize / 2, 0, 2 * Math.PI);
            drawCtx.fill();
            drawCtx.restore();

            lastAirPoint = { x: indexTip.x, y: indexTip.y };
        }
    });

    if (!isPointing) {
        lastAirPoint = null;
    }
}

// --- MODEL INITIALIZATION ---
async function initAIModels() {
    try {
        elements.systemStatus.textContent = 'Syncing...';
        elements.systemStatus.className = 'badge-val';

        await tf.ready();
        elements.backendType.textContent = tf.getBackend().toUpperCase();

        cocoModel = await cocoSsd.load({ base: 'mobilenet_v1' });
        document.getElementById('chk-coco').classList.add('loaded');

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        handDetector = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });
        const chkHands = document.getElementById('chk-hands');
        if (chkHands) chkHands.classList.add('loaded');

        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        document.getElementById('chk-face').classList.add('loaded');

        await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
        document.getElementById('chk-landmarks').classList.add('loaded');

        await faceapi.nets.faceExpressionNet.loadFromUri('/models');
        document.getElementById('chk-emotions').classList.add('loaded');

        await faceapi.nets.ageGenderNet.loadFromUri('/models');
        document.getElementById('chk-age').classList.add('loaded');

        modelsLoaded = true;
        elements.modelLoader.classList.add('hidden');
        elements.systemStatus.textContent = 'Active';
        elements.systemStatus.className = 'badge-val pulse-green';

        showToast('AI networks initialized successfully.', 'success');
        await startCamera();
    } catch (error) {
        console.error('Model loading failed:', error);
        elements.systemStatus.textContent = 'Failed';
        showError('Failed loading machine learning models: ' + error.message);
    }
}

// --- WEBCAM MANAGEMENT ---
async function setupCameraDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        elements.cameraSelect.innerHTML = '';

        if (videoDevices.length === 0) {
            elements.cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            return;
        }

        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            elements.cameraSelect.appendChild(option);
        });

        if (!currentDeviceId && videoDevices.length > 0) {
            currentDeviceId = videoDevices[0].deviceId;
            elements.cameraSelect.value = currentDeviceId;
        }
    } catch (err) {
        console.warn('Unable to list camera devices:', err);
    }
}

async function startCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    elements.errorBanner.classList.add('hidden');
    const constraints = {
        video: {
            deviceId: currentDeviceId ? { exact: currentDeviceId } : undefined,
            width: { ideal: 640 },
            height: { ideal: 480 }
        },
        audio: false
    };

    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.video.srcObject = stream;

        elements.video.onloadedmetadata = () => {
            elements.video.play();
            const w = elements.video.videoWidth;
            const h = elements.video.videoHeight;
            elements.resolution.textContent = `${w}x${h}`;
            elements.canvas.width = w;
            elements.canvas.height = h;
            if (elements.drawingCanvas) {
                elements.drawingCanvas.width = w;
                elements.drawingCanvas.height = h;
            }

            setupCameraDevices();

            if (inferenceRequestId) cancelAnimationFrame(inferenceRequestId);
            isPaused = false;
            elements.btnPause.classList.remove('active');
            elements.btnPause.innerHTML = `<svg viewBox="0 0 24 24" class="icon" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;

            drawLoop();
        };
    } catch (err) {
        console.error('Camera startup failed:', err);
        showError('Webcam access was denied or device is unavailable. Please check browser permissions.');
    }
}

// --- ANIMATED PARTICLE SYSTEMS ---
class FingertipParticle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 1.5;
        this.vy = -Math.random() * 2 - 0.5;
        this.color = color;
        this.alpha = 1.0;
        this.size = Math.random() * 2.5 + 1.5;
        this.decay = Math.random() * 0.02 + 0.015;
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.alpha -= this.decay;
    }
    draw(ctx) {
        if (this.alpha <= 0) return;
        ctx.save();
        ctx.globalAlpha = this.alpha;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.restore();
    }
}

class WebBlastEffect {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.radius = 5;
        this.maxRadius = 130;
        this.alpha = 1.0;
        this.speed = 3.5;
        this.lines = [];
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2 + (Math.random() - 0.5) * 0.25;
            this.lines.push({
                angle: angle,
                length: Math.random() * 70 + 60
            });
        }
    }
    update() {
        this.radius += this.speed;
        this.alpha = Math.max(0, 1 - (this.radius / this.maxRadius));
    }
    draw(ctx) {
        if (this.alpha <= 0) return;
        ctx.save();
        ctx.globalAlpha = this.alpha;

        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 3;
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 15;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 0.6, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        this.lines.forEach(line => {
            const currentLen = (this.radius / this.maxRadius) * line.length;
            const endX = this.x + Math.cos(line.angle) * currentLen;
            const endY = this.y + Math.sin(line.angle) * currentLen;

            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1.8;
            ctx.stroke();

            const nextAngle = line.angle + (Math.PI * 2 / 8);
            const nextEndX = this.x + Math.cos(nextAngle) * currentLen * 0.8;
            const nextEndY = this.y + Math.sin(nextAngle) * currentLen * 0.8;
            ctx.beginPath();
            ctx.moveTo(endX, endY);
            ctx.quadraticCurveTo((endX + nextEndX) / 2, (endY + nextEndY) / 2, nextEndX, nextEndY);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.lineWidth = 0.8;
            ctx.stroke();
        });

        ctx.restore();
    }
}

const activeParticles = [];
const activeWebBlasts = [];
const lastBlastTimes = { Left: 0, Right: 0 };

function showError(msg) {
    elements.errorMessage.textContent = msg;
    elements.errorBanner.classList.remove('hidden');
}

// --- DETECTION & CANVAS LOOP ---
async function drawLoop() {
    if (isPaused) return;

    frameCount++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
        fps = Math.round((frameCount * 1000) / (now - fpsTimer));
        elements.fpsCounter.textContent = fps.toString().padStart(2, '0');
        frameCount = 0;
        fpsTimer = now;
    }

    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    // Update and draw active particles
    for (let i = activeParticles.length - 1; i >= 0; i--) {
        const p = activeParticles[i];
        p.update();
        p.draw(ctx);
        if (p.alpha <= 0) activeParticles.splice(i, 1);
    }

    // Update and draw active web blasts
    for (let i = activeWebBlasts.length - 1; i >= 0; i--) {
        const b = activeWebBlasts[i];
        b.update();
        b.draw(ctx);
        if (b.alpha <= 0) activeWebBlasts.splice(i, 1);
    }

    let detectedObjects = [];
    let visibleTrackedFaces = [];
    let handDetections = null;

    try {
        if (modelsLoaded) {
            const tfPromises = [];

            if (detectObjectsEnabled && cocoModel) {
                preprocessFrame(elements.video);
                tfPromises.push(cocoModel.detect(preprocessCanvas));
            } else {
                tfPromises.push(Promise.resolve([]));
            }

            if (detectFacesEnabled) {
                const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.2 });
                tfPromises.push(
                    faceapi.detectAllFaces(elements.video, options)
                        .withFaceLandmarks()
                        .withFaceExpressions()
                        .withAgeAndGender()
                );
            } else {
                tfPromises.push(Promise.resolve([]));
            }

            const [objDetections, faceDetections] = await Promise.all(tfPromises);

            if (detectHandsEnabled && handDetector) {
                try {
                    handDetections = handDetector.detectForVideo(elements.video, performance.now());
                } catch (handErr) {
                    console.warn('Hand estimation error (skipping frame):', handErr.message || handErr);
                    handDetections = null;
                }
            }

            detectionHistory.push(objDetections);
            if (detectionHistory.length > DETECTION_HISTORY_FRAMES) {
                detectionHistory.shift();
            }
            const smoothedDetections = smoothDetections(detectionHistory);
            const rawDetectedObjects = smoothedDetections.filter(d => d.score >= confidenceThreshold);
            detectedObjects = objectTracker.update(rawDetectedObjects);
            visibleTrackedFaces = faceTracker.update(faceDetections);
            processDrowsinessAndAttention(visibleTrackedFaces);
        }
    } catch (err) {
        console.error('Inference loop error:', err);
    }

    let mappedHands = [];
    if (detectHandsEnabled && handDetections && handDetections.landmarks && handDetections.landmarks.length > 0) {
        mappedHands = handDetections.landmarks.map((landmarks, idx) => {
            const keypoints = landmarks.map(pt => ({
                x: pt.x * elements.canvas.width,
                y: pt.y * elements.canvas.height,
                z: pt.z
            }));
            const rawHandedness = handDetections.handedness[idx]?.[0]?.categoryName || 'Left';
            const isLeft = rawHandedness === 'Left';
            return { keypoints, isLeft, handedness: rawHandedness };
        });
        processAirCanvas(mappedHands);
    } else {
        processAirCanvas([]);
    }

    if (hudActive) {
        const objectColor = getThemeColor('--accent-primary', '#2563eb');
        const faceColor = getThemeColor('--accent-primary', '#4f46e5');

        detectedObjects.forEach(obj => {
            const [x, y, w, h] = obj.bbox;

            ctx.strokeStyle = objectColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = 'rgba(37, 99, 235, 0.04)';
            ctx.fillRect(x, y, w, h);

            const text = `${obj.class} #${obj.id} (${Math.round(obj.score * 100)}%)`;
            drawSassTextBadge(ctx, text, x, y, objectColor);
        });

        visibleTrackedFaces.forEach(face => {
            const box = face.box;
            const x = box.x;
            const y = box.y;
            const w = box.width;
            const h = box.height;

            const strokeColor = isDrowsyAlarmActive ? '#ef4444' : faceColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = isDrowsyAlarmActive ? 4 : 2;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = isDrowsyAlarmActive ? 'rgba(239, 68, 68, 0.15)' : 'rgba(79, 70, 229, 0.04)';
            ctx.fillRect(x, y, w, h);

            const roundedAge = Math.round(face.age);
            const genderSymbol = face.gender === 'male' ? 'M' : 'F';
            const dominantEmotion = getDominantEmotion(face.expressions);
            const earStr = currentEarValue ? currentEarValue.toFixed(2) : '0.30';
            const label = isDrowsyAlarmActive ? 
                `⚠️ Person #${face.id} | DROWSY ALARM!` : 
                `Person #${face.id} | ${genderSymbol} (~${roundedAge}) | ${dominantEmotion} | EAR: ${earStr}`;

            drawSassTextBadge(ctx, label, x, y, strokeColor);

            if (face.landmarks && face.landmarks.positions) {
                ctx.fillStyle = 'rgba(59, 130, 246, 0.6)';
                face.landmarks.positions.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 1.5, 0, 2 * Math.PI);
                    ctx.fill();
                });
            }
        });

        if (mappedHands.length > 0) {

            let handsClose = false;
            let closeDist = 0;
            if (mappedHands.length >= 2) {
                const h1 = mappedHands[0];
                const h2 = mappedHands[1];
                const p1 = h1.keypoints[9];
                const p2 = h2.keypoints[9];
                closeDist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                if (closeDist < 350) {
                    handsClose = true;
                }
            }

            mappedHands.forEach(hand => {
                const keypoints = hand.keypoints;
                const getPt = (i) => keypoints[i];
                const palmCenter = getPt(9);
                const isLeft = hand.isLeft;
                const gesture = classifyGesture(keypoints);
                const fingertips = [4, 8, 12, 16, 20];
                
                const glowHue = isLeft ? '#ff007f' : '#00f2fe';
                const nodeColor = isLeft ? '#ff80bf' : '#80f2ff';

                ctx.save();

                if (spiderEffectEnabled) {
                    const SKELETON_CONNECTIONS = [
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20],
                        [5, 9], [9, 13], [13, 17]
                    ];

                    SKELETON_CONNECTIONS.forEach(([j1, j2]) => {
                        const p1 = getPt(j1);
                        const p2 = getPt(j2);
                        ctx.save();
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = glowHue;
                        ctx.lineWidth = 2.2;
                        ctx.setLineDash([5, 5]);
                        ctx.lineDashOffset = -Math.floor(Date.now() / 25) % 10;
                        ctx.shadowColor = glowHue;
                        ctx.shadowBlur = 10;
                        ctx.stroke();
                        ctx.restore();
                    });

                    const webConnections = [
                        [2, 5], [5, 9], [9, 13], [13, 17],
                        [3, 6], [6, 10], [10, 14], [14, 18],
                        [3, 7], [7, 11], [11, 15], [15, 19],
                        [4, 8], [8, 12], [12, 16], [16, 20]
                    ];

                    webConnections.forEach(([j1, j2]) => {
                        const p1 = getPt(j1);
                        const p2 = getPt(j2);
                        ctx.save();
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        const mx = (p1.x + p2.x) / 2;
                        const my = (p1.y + p2.y) / 2;
                        const wrist = getPt(0);
                        const cpx = mx + (wrist.x - mx) * 0.12;
                        const cpy = my + (wrist.y - my) * 0.12;
                        ctx.quadraticCurveTo(cpx, cpy, p2.x, p2.y);
                        ctx.strokeStyle = isLeft ? 'rgba(255, 0, 127, 0.28)' : 'rgba(0, 242, 254, 0.28)';
                        ctx.lineWidth = 0.9;
                        ctx.shadowColor = glowHue;
                        ctx.shadowBlur = 3;
                        ctx.stroke();
                        ctx.restore();
                    });

                    if (Math.random() < 0.25) {
                        fingertips.forEach(tipIdx => {
                            const pt = getPt(tipIdx);
                            activeParticles.push(new FingertipParticle(pt.x, pt.y, glowHue));
                        });
                    }

                    if (gesture.name.includes('WEB SHOOTER')) {
                        const handSide = isLeft ? 'Left' : 'Right';
                        if (Date.now() - lastBlastTimes[handSide] > 600) {
                            activeWebBlasts.push(new WebBlastEffect(palmCenter.x, palmCenter.y, glowHue));
                            lastBlastTimes[handSide] = Date.now();
                            playWebShooterSFX();
                            speakVoiceAlert('Web shooter active');
                            showToast(`${handSide.toUpperCase()} HAND: Web Shooter Activated!`, 'success');
                        }
                    } else if (['OK SIGN', 'PEACE SIGN', 'THUMBS UP'].some(g => gesture.name.includes(g))) {
                        playGestureChimeSFX();
                    }
                }

                keypoints.forEach((pt, idx) => {
                    const pulse = Math.sin(Date.now() / 150 + idx) * 2;
                    const size = 5.5 + pulse;
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, size, 0, Math.PI * 2);
                    ctx.fillStyle = '#ffffff';
                    ctx.shadowColor = glowHue;
                    ctx.shadowBlur = 14;
                    ctx.fill();
                    ctx.restore();

                    if (showJointNumbersEnabled) {
                        ctx.save();
                        ctx.translate(pt.x, pt.y);
                        ctx.scale(-1, 1);
                        ctx.font = 'bold 9px JetBrains Mono, monospace';
                        ctx.fillStyle = '#ffffff';
                        ctx.fillText(idx, 8, 3);
                        ctx.restore();
                    }
                });

                ctx.restore();

                const wristPt = getPt(0);
                const handSide = isLeft ? 'LEFT HAND' : 'RIGHT HAND';
                const handLabel = `${handSide} | ${gesture.name}`;
                if (detectGesturesEnabled) {
                    drawHandBadge(ctx, handLabel, wristPt.x, wristPt.y + 28, gesture.color);
                }

                if (!handsClose && spiderEffectEnabled) {
                    if (isLeft) {
                        let nearestFace = null;
                        let minFaceDist = Infinity;
                        visibleTrackedFaces.forEach(face => {
                            const fx = face.box.x + face.box.width / 2;
                            const fy = face.box.y + face.box.height / 2;
                            const dist = Math.hypot(palmCenter.x - fx, palmCenter.y - fy);
                            if (dist < minFaceDist) {
                                  minFaceDist = dist;
                                  nearestFace = { x: fx, y: fy };
                            }
                        });

                        if (nearestFace) {
                            ctx.save();
                            ctx.beginPath();
                            ctx.moveTo(palmCenter.x, palmCenter.y);
                            ctx.quadraticCurveTo(
                                (palmCenter.x + nearestFace.x) / 2 + Math.sin(Date.now() / 200) * 20,
                                (palmCenter.y + nearestFace.y) / 2 + Math.cos(Date.now() / 200) * 20,
                                nearestFace.x, nearestFace.y
                            );
                            ctx.strokeStyle = 'rgba(255, 0, 127, 0.7)';
                            ctx.lineWidth = 2;
                            ctx.setLineDash([4, 6]);
                            ctx.shadowColor = '#ff007f';
                            ctx.shadowBlur = 8;
                            ctx.stroke();
                            ctx.restore();

                            const mx = (palmCenter.x + nearestFace.x) / 2;
                            const my = (palmCenter.y + nearestFace.y) / 2;
                            drawSassTextBadge(ctx, `BIO-LINK | DIST: ${Math.round(minFaceDist)}px`, mx, my - 10, '#ff007f');
                        }
                    } else {
                        let nearestObj = null;
                        let minObjDist = Infinity;
                        detectedObjects.forEach(obj => {
                            const [ox, oy, ow, oh] = obj.bbox;
                            const ocx = ox + ow / 2;
                            const ocy = oy + oh / 2;
                            const dist = Math.hypot(palmCenter.x - ocx, palmCenter.y - ocy);
                            if (dist < minObjDist) {
                                  minObjDist = dist;
                                  nearestObj = { x: ocx, y: ocy, name: obj.class };
                            }
                        });

                        if (nearestObj) {
                            ctx.save();
                            ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
                            ctx.lineWidth = 1.5;
                            ctx.shadowColor = '#00f2fe';
                            ctx.shadowBlur = 8;

                            ctx.beginPath();
                            const segments = 25;
                            for (let i = 0; i <= segments; i++) {
                                const t = i / segments;
                                const lx = palmCenter.x + (nearestObj.x - palmCenter.x) * t;
                                const ly = palmCenter.y + (nearestObj.y - palmCenter.y) * t;
                                const helixWave = Math.sin(t * Math.PI * 6 - Date.now() / 80) * 15;
                                const nx = -(nearestObj.y - palmCenter.y);
                                const ny = nearestObj.x - palmCenter.x;
                                const nlen = Math.hypot(nx, ny) || 1;
                                ctx.lineTo(lx + (nx / nlen) * helixWave, ly + (ny / nlen) * helixWave);
                            }
                            ctx.stroke();
                            ctx.restore();

                            const mx = (palmCenter.x + nearestObj.x) / 2;
                            const my = (palmCenter.y + nearestObj.y) / 2;
                            drawSassTextBadge(ctx, `TETHER: ${nearestObj.name.toUpperCase()} | DIST: ${Math.round(minObjDist)}px`, mx, my - 10, '#00f2fe');
                        }
                    }
                }
            });

            if (handsClose && mappedHands.length >= 2 && spiderEffectEnabled) {
                const hand1 = mappedHands[0];
                const hand2 = mappedHands[1];
                const kps1 = hand1.keypoints;
                const kps2 = hand2.keypoints;

                if (kps1 && kps2 && kps1.length >= 21 && kps2.length >= 21) {
                    const p1 = kps1[9];
                    const p2 = kps2[9];
                    const distance = closeDist;
                    const threshold = 350;

                    if (distance < threshold) {
                        const proximityFactor = Math.max(0, 1 - (distance / threshold));
                        ctx.save();

                        const midX = (p1.x + p2.x) / 2;
                        const midY = (p1.y + p2.y) / 2;

                        const tips = [4, 8, 12, 16, 20];
                        tips.forEach(tip => {
                            const tip1 = kps1[tip];
                            const tip2 = kps2[tip];

                            ctx.beginPath();
                            ctx.moveTo(tip1.x, tip1.y);

                            const ctrlX = midX + (Math.sin(Date.now() / 200 + tip) * 15);
                            const ctrlY = midY + (Math.cos(Date.now() / 200 + tip) * 15);
                            ctx.quadraticCurveTo(ctrlX, ctrlY, tip2.x, tip2.y);

                            const gradient = ctx.createLinearGradient(tip1.x, tip1.y, tip2.x, tip2.y);
                            gradient.addColorStop(0, 'rgba(0, 242, 254, ' + (0.2 + proximityFactor * 0.6) + ')');
                            gradient.addColorStop(0.5, 'rgba(255, 0, 127, ' + (0.4 + proximityFactor * 0.6) + ')');
                            gradient.addColorStop(1, 'rgba(255, 0, 127, ' + (0.2 + proximityFactor * 0.6) + ')');

                            ctx.strokeStyle = gradient;
                            ctx.lineWidth = 1 + proximityFactor * 2;
                            ctx.shadowColor = '#ff007f';
                            ctx.shadowBlur = 5 + proximityFactor * 10;
                            ctx.stroke();
                        });

                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        const segments = 15;
                        for (let i = 1; i <= segments; i++) {
                            const t = i / segments;
                            const x = p1.x + (p2.x - p1.x) * t;
                            const y = p1.y + (p2.y - p1.y) * t;
                            if (i < segments) {
                                const wave = Math.sin(t * Math.PI) * (Math.sin(Date.now() / 60 + t * 12) * (10 + proximityFactor * 15));
                                const nx = -(p2.y - p1.y);
                                const ny = p2.x - p1.x;
                                const len = Math.hypot(nx, ny) || 1;
                                const ox = (nx / len) * wave;
                                const oy = (ny / len) * wave;
                                ctx.lineTo(x + ox, y + oy);
                            } else {
                                ctx.lineTo(p2.x, p2.y);
                            }
                        }
                        ctx.strokeStyle = 'rgba(0, 242, 254, ' + (0.4 + proximityFactor * 0.6) + ')';
                        ctx.lineWidth = 1.5 + proximityFactor * 3;
                        ctx.shadowColor = '#00f2fe';
                        ctx.shadowBlur = 10 + proximityFactor * 15;
                        ctx.stroke();

                        const coreRadius = 15 + proximityFactor * 25;
                        const pulse = Math.sin(Date.now() / 100) * 5;

                        const radGrad = ctx.createRadialGradient(midX, midY, 2, midX, midY, coreRadius + 10);
                        radGrad.addColorStop(0, 'rgba(255, 0, 127, ' + (0.6 + proximityFactor * 0.4) + ')');
                        radGrad.addColorStop(0.3, 'rgba(0, 242, 254, ' + (0.4 + proximityFactor * 0.3) + ')');
                        radGrad.addColorStop(1, 'rgba(0, 242, 254, 0)');

                        ctx.fillStyle = radGrad;
                        ctx.beginPath();
                        ctx.arc(midX, midY, coreRadius + 10 + pulse, 0, 2 * Math.PI);
                        ctx.fill();

                        ctx.fillStyle = '#ffffff';
                        ctx.shadowColor = '#ff007f';
                        ctx.shadowBlur = 15;
                        ctx.beginPath();
                        ctx.arc(midX, midY, 4 + proximityFactor * 8, 0, 2 * Math.PI);
                        ctx.fill();

                        ctx.restore();

                        const energyLevel = Math.round(proximityFactor * 100);
                        const badgeColor = proximityFactor > 0.75 ? '#ff007f' : '#00f2fe';
                        drawSassTextBadge(
                            ctx,
                            `CYBER LINK | DIST: ${Math.round(distance)}px | POWER: ${energyLevel}%`,
                            midX,
                            midY - coreRadius - 18,
                            badgeColor
                        );
                    }
                }
            }
        }
    }

    const nowTime = Date.now();
    if (nowTime - lastDomUpdateTime >= 200) {
        lastDomUpdateTime = nowTime;
        if (activeTab === 'tab-detections') {
            updateActiveDetectionsLog(detectedObjects);
        } else if (activeTab === 'tab-biometrics') {
            updateFaceAnalyticsSidebar(visibleTrackedFaces);
        } else if (activeTab === 'tab-stats') {
            updateSessionTelemetry(visibleTrackedFaces);
        }
    }

    if (elements.tabObjCount) {
        elements.tabObjCount.textContent = detectedObjects.length;
    }
    if (elements.tabFaceCount) {
        elements.tabFaceCount.textContent = visibleTrackedFaces.length;
    }

    inferenceRequestId = requestAnimationFrame(drawLoop);
}

// --- GESTURE RECOGNITION ---
function classifyGesture(kp) {
    const fingerExtended = (tipIdx, pipIdx) => kp[tipIdx].y < kp[pipIdx].y - 8;

    const indexUp = fingerExtended(8, 6);
    const middleUp = fingerExtended(12, 10);
    const ringUp = fingerExtended(16, 14);
    const pinkyUp = fingerExtended(20, 18);

    const thumbTip = kp[4];
    const thumbIp = kp[3];
    const indexMcp = kp[5];
    const thumbDistance = Math.hypot(thumbTip.x - indexMcp.x, thumbTip.y - indexMcp.y);
    const thumbIPDistance = Math.hypot(thumbIp.x - indexMcp.x, thumbIp.y - indexMcp.y);
    const thumbExtended = thumbDistance > thumbIPDistance * 1.15;

    const indexThumbDist = Math.hypot(kp[4].x - kp[8].x, kp[4].y - kp[8].y);
    const isOK = indexThumbDist < 30 && middleUp && ringUp && pinkyUp;

    const isWebShooter = thumbExtended && indexUp && !middleUp && !ringUp && pinkyUp;

    const allFingersFolded = !indexUp && !middleUp && !ringUp && !pinkyUp;

    if (isWebShooter) {
        return { name: '🕸️ WEB SHOOTER', color: '#00f2fe' };
    }
    if (isOK) {
        return { name: '👌 OK SIGN', color: '#e040fb' };
    }
    if (thumbExtended && allFingersFolded) {
        if (kp[4].y < kp[2].y) {
            return { name: '👍 THUMBS UP', color: '#69f0ae' };
        } else {
            return { name: '👎 THUMBS DOWN', color: '#ff5252' };
        }
    }
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
        return { name: '✌️ PEACE SIGN', color: '#00e5ff' };
    }
    if (indexUp && !middleUp && !ringUp && !pinkyUp) {
        return { name: '☝️ POINTING', color: '#ff9800' };
    }
    if (indexUp && middleUp && ringUp && pinkyUp && thumbExtended) {
        return { name: '🖐️ OPEN PALM', color: '#b39ddb' };
    }
    if (allFingersFolded && !thumbExtended) {
        return { name: '✊ CLOSED FIST', color: '#ef5350' };
    }

    return { name: '🤚 HAND', color: '#90a4ae' };
}

function drawHandBadge(ctx, text, x, y, color) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(-1, 1);

    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(text).width;
    const pad = 8;
    const bw = tw + pad * 2;
    const bh = 22;
    const br = 5;

    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    roundRect(ctx, -bw / 2, -bh, bw, bh, br);
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.fillText(text, 0, -bh / 2);
    ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// --- CORE INTERFERENCE PREPROCESSING ---
function preprocessFrame(videoEl) {
    const w = videoEl.videoWidth;
    const h = videoEl.videoHeight;
    if (w === 0 || h === 0) return;
    preprocessCanvas.width = w;
    preprocessCanvas.height = h;
    preprocessCtx.filter = 'contrast(1.2) brightness(1.1)';
    preprocessCtx.drawImage(videoEl, 0, 0, w, h);
    preprocessCtx.filter = 'none';
}

function smoothDetections(history) {
    if (history.length === 0) return [];
    const latest = history[history.length - 1];
    if (history.length === 1) return latest;

    return latest.map(det => {
        let totalScore = det.score;
        let count = 1;

        for (let f = 0; f < history.length - 1; f++) {
            for (const prev of history[f]) {
                if (prev.class !== det.class) continue;
                const b1 = det.bbox;
                const b2 = prev.bbox;
                const xA = Math.max(b1[0], b2[0]);
                const yA = Math.max(b1[1], b2[1]);
                const xB = Math.min(b1[0] + b1[2], b2[0] + b2[2]);
                const yB = Math.min(b1[1] + b1[3], b2[1] + b2[3]);
                const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
                if (inter > 0) {
                    const union = b1[2] * b1[3] + b2[2] * b2[3] - inter;
                    if (inter / union > 0.3) {
                        totalScore += prev.score;
                        count++;
                    }
                }
            }
        }

        return { ...det, score: totalScore / count };
    });
}

function drawSassTextBadge(ctx, text, x, y, badgeColor) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(-1, 1);

    ctx.font = '500 11px Inter, system-ui, sans-serif';
    const textWidth = ctx.measureText(text).width;
    const padding = 6;
    const badgeWidth = textWidth + (padding * 2);
    const badgeHeight = 18;
    const badgeRadius = 4;

    ctx.fillStyle = badgeColor;
    drawRoundedRect(ctx, -badgeWidth, -14 - badgeHeight, badgeWidth, badgeHeight, badgeRadius);

    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, -badgeWidth + padding, -14 - (badgeHeight / 2));

    ctx.restore();
}

function drawRoundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
}

function getDominantEmotion(expressions) {
    if (!expressions) return 'neutral';
    let maxVal = 0;
    let dominant = 'neutral';
    for (const [emo, val] of Object.entries(expressions)) {
        if (val > maxVal) {
            maxVal = val;
            dominant = emo;
        }
    }
    return dominant;
}

// --- SIDEBAR DATA RENDER ---
function updateActiveDetectionsLog(objects) {
    elements.totalObjectsBadge.textContent = objects.length;
    if (objects.length === 0) {
        elements.detectionsContainer.innerHTML = `
      <div class="empty-state">
        <p>No objects detected above threshold.</p>
      </div>`;
        return;
    }

    elements.detectionsContainer.innerHTML = objects.map(obj => {
        const scorePct = Math.round(obj.score * 100);
        return `
      <div class="detection-row">
        <span class="detection-label">${obj.class} #${obj.id}</span>
        <div class="detection-bar-wrap">
          <div class="detection-bar-fill" style="width: ${scorePct}%"></div>
        </div>
        <span class="detection-pct text-mono">${scorePct}%</span>
      </div>`;
    }).join('');
}

function updateFaceAnalyticsSidebar(faces) {
    elements.faceCountBadge.textContent = faces.length;
    if (faces.length === 0) {
        elements.faceAnalyticsContainer.innerHTML = `
      <div class="empty-state">
        <p>No active faces detected.</p>
      </div>`;
        return;
    }

    elements.faceAnalyticsContainer.innerHTML = faces.map(face => {
        const expressions = face.expressions || {};
        const age = Math.round(face.age);
        const dominantEmotion = getDominantEmotion(expressions);
        const emotionProb = expressions[dominantEmotion] ? Math.round(expressions[dominantEmotion] * 100) : 0;

        let badgeClass = 'emotion-badge-neutral';
        if (['happy', 'neutral', 'sad', 'angry', 'surprised', 'fearful', 'disgusted'].includes(dominantEmotion)) {
            badgeClass = `emotion-badge-${dominantEmotion}`;
        }

        return `
      <div class="face-profile-card">
        <div class="face-summary-row">
          <span class="face-id-tag text-mono">Person #${face.id}</span>
          <span class="face-age-gender">${face.gender.toUpperCase()} / ~${age}y</span>
          <span class="face-emotion-badge ${badgeClass}">${dominantEmotion} ${emotionProb}%</span>
        </div>
        <div class="face-details-grid">
          <div class="detail-item">
            <span class="detail-lbl">Happy</span>
            <span class="detail-val text-mono">${Math.round((expressions.happy || 0) * 100)}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-lbl">Neutral</span>
            <span class="detail-val text-mono">${Math.round((expressions.neutral || 0) * 100)}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-lbl">Surprised</span>
            <span class="detail-val text-mono">${Math.round((expressions.surprised || 0) * 100)}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-lbl">Sad / Angry</span>
            <span class="detail-val text-mono">${Math.round((Math.max(expressions.sad || 0, expressions.angry || 0)) * 100)}%</span>
          </div>
        </div>
      </div>`;
    }).join('');
}

function updateSessionTelemetry(faces) {
    elements.telUniqueFaces.textContent = sessionStats.uniqueFacesCount;

    const smoothedAges = [];
    for (const [id, ageArr] of Object.entries(sessionStats.trackedAges)) {
        if (ageArr.length > 0) {
            const avg = ageArr.reduce((a, b) => a + b, 0) / ageArr.length;
            smoothedAges.push(avg);
        }
    }

    if (smoothedAges.length > 0) {
        const totalAge = smoothedAges.reduce((a, b) => a + b, 0);
        const avgAge = Math.round(totalAge / smoothedAges.length);
        elements.telAvgAge.textContent = `${avgAge}y`;
    } else {
        elements.telAvgAge.textContent = '--';
    }

    // Attention & Drowsiness Telemetry Update
    if (elements.telBlinkCount) elements.telBlinkCount.textContent = sessionBlinkCount;
    if (elements.telEarVal) elements.telEarVal.textContent = currentEarValue ? currentEarValue.toFixed(2) : '0.30';
    if (elements.barEar) {
        const earPct = Math.min(100, Math.max(0, Math.round((currentEarValue / 0.35) * 100)));
        elements.barEar.style.width = `${earPct}%`;
    }
    if (elements.telAttentionVal) elements.telAttentionVal.textContent = `${currentAttentionIndex}%`;
    if (elements.barAttention) elements.barAttention.style.width = `${currentAttentionIndex}%`;

    if (elements.telDrowsyStatus) {
        if (isDrowsyAlarmActive) {
            elements.telDrowsyStatus.textContent = '🚨 DROWSY ALARM';
            elements.telDrowsyStatus.className = 'telemetry-value badge-val-sm pulse-red';
        } else if (isEyeClosedState) {
            elements.telDrowsyStatus.textContent = 'EYES CLOSED';
            elements.telDrowsyStatus.className = 'telemetry-value badge-val-sm pulse-yellow';
        } else {
            elements.telDrowsyStatus.textContent = 'NORMAL';
            elements.telDrowsyStatus.className = 'telemetry-value badge-val-sm pulse-green';
        }
    }

    faces.forEach(face => {
        const dominant = getDominantEmotion(face.expressions);
        if (dominant in sessionStats.emotionHits) {
            sessionStats.emotionHits[dominant]++;
            sessionStats.emotionTotal++;
        }
    });

    if (sessionStats.emotionTotal > 0) {
        const updateBar = (bar, label, key) => {
            const pct = Math.round((sessionStats.emotionHits[key] / sessionStats.emotionTotal) * 100);
            bar.style.width = `${pct}%`;
            label.textContent = `${pct}%`;
        };

        updateBar(elements.barHappy, elements.pctHappy, 'happy');
        updateBar(elements.barNeutral, elements.pctNeutral, 'neutral');
        updateBar(elements.barSurprised, elements.pctSurprised, 'surprised');
        updateBar(elements.barSad, elements.pctSad, 'sad');
        updateBar(elements.barAngry, elements.pctAngry, 'angry');
    }
}

// --- CONTROLS ACTIONS ---
function toggleInferencePause() {
    isPaused = !isPaused;
    if (isPaused) {
        if (inferenceRequestId) cancelAnimationFrame(inferenceRequestId);
        elements.btnPause.classList.add('active');
        elements.btnPause.innerHTML = `<svg viewBox="0 0 24 24" class="icon" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
        elements.systemStatus.textContent = 'Paused';
        showToast('AI analysis stream paused.');
    } else {
        elements.btnPause.classList.remove('active');
        elements.btnPause.innerHTML = `<svg viewBox="0 0 24 24" class="icon" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
        elements.systemStatus.textContent = 'Active';
        showToast('AI analysis stream resumed.', 'success');
        drawLoop();
    }
}

function captureSnapshot() {
    try {
        const bufferCanvas = document.createElement('canvas');
        bufferCanvas.width = elements.canvas.width;
        bufferCanvas.height = elements.canvas.height;
        const bufferCtx = bufferCanvas.getContext('2d');

        bufferCtx.save();
        bufferCtx.translate(bufferCanvas.width, 0);
        bufferCtx.scale(-1, 1);
        bufferCtx.drawImage(elements.video, 0, 0, bufferCanvas.width, bufferCanvas.height);
        bufferCtx.restore();

        if (elements.drawingCanvas) {
            bufferCtx.save();
            bufferCtx.translate(bufferCanvas.width, 0);
            bufferCtx.scale(-1, 1);
            bufferCtx.drawImage(elements.drawingCanvas, 0, 0, bufferCanvas.width, bufferCanvas.height);
            bufferCtx.restore();
        }

        if (hudActive) {
            bufferCtx.save();
            bufferCtx.translate(bufferCanvas.width, 0);
            bufferCtx.scale(-1, 1);
            bufferCtx.drawImage(elements.canvas, 0, 0, bufferCanvas.width, bufferCanvas.height);
            bufferCtx.restore();
        }

        const dataUrl = bufferCanvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `AetherVision-Capture-${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
        showToast('Telemetry screenshot exported.', 'success');
    } catch (err) {
        console.error('Screenshot failed:', err);
        showToast('Screenshot capture failed.', 'error');
    }
}

// --- AIR CANVAS TOOLBAR EVENT LISTENERS ---
if (elements.btnToggleAir) {
    elements.btnToggleAir.addEventListener('click', () => {
        airCanvasEnabled = !airCanvasEnabled;
        elements.btnToggleAir.classList.toggle('active', airCanvasEnabled);
        showToast(`Air Canvas ${airCanvasEnabled ? 'enabled' : 'disabled'}.`);
    });
}

if (elements.colorSwatches) {
    elements.colorSwatches.forEach(swatch => {
        swatch.addEventListener('click', () => {
            elements.colorSwatches.forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            airBrushColor = swatch.dataset.color || '#00f2fe';
        });
    });
}

if (elements.brushSizeSlider) {
    elements.brushSizeSlider.addEventListener('input', (e) => {
        airBrushSize = parseInt(e.target.value);
    });
}

if (elements.btnClearCanvas) {
    elements.btnClearCanvas.addEventListener('click', clearAirCanvas);
}

// --- MOUSE & TOUCH FALLBACK DRAWING LISTENERS ---
let isMouseDrawing = false;
let mouseLastPoint = null;

if (elements.drawingCanvas) {
    const getCanvasPos = (e) => {
        const rect = elements.drawingCanvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const rawX = clientX - rect.left;
        const flippedX = rect.width - rawX;
        const scaleX = elements.drawingCanvas.width / rect.width;
        const scaleY = elements.drawingCanvas.height / rect.height;
        return {
            x: flippedX * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    };

    const startDraw = (e) => {
        if (!airCanvasEnabled) return;
        isMouseDrawing = true;
        mouseLastPoint = getCanvasPos(e);
    };

    const drawMove = (e) => {
        if (!isMouseDrawing || !airCanvasEnabled || !mouseLastPoint) return;
        const currentPos = getCanvasPos(e);
        const drawCtx = elements.drawingCanvas.getContext('2d');

        drawCtx.save();
        drawCtx.strokeStyle = airBrushColor;
        drawCtx.fillStyle = airBrushColor;
        drawCtx.lineWidth = airBrushSize;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.shadowColor = airBrushColor;
        drawCtx.shadowBlur = airBrushSize * 1.5;

        drawCtx.beginPath();
        drawCtx.moveTo(mouseLastPoint.x, mouseLastPoint.y);
        drawCtx.lineTo(currentPos.x, currentPos.y);
        drawCtx.stroke();
        drawCtx.restore();

        mouseLastPoint = currentPos;
    };

    const stopDraw = () => {
        isMouseDrawing = false;
        mouseLastPoint = null;
    };

    elements.drawingCanvas.addEventListener('mousedown', startDraw);
    elements.drawingCanvas.addEventListener('mousemove', drawMove);
    elements.drawingCanvas.addEventListener('mouseup', stopDraw);
    elements.drawingCanvas.addEventListener('mouseleave', stopDraw);

    elements.drawingCanvas.addEventListener('touchstart', startDraw, { passive: true });
    elements.drawingCanvas.addEventListener('touchmove', drawMove, { passive: true });
    elements.drawingCanvas.addEventListener('touchend', stopDraw);
}

// --- MASTER FEATURE CONTROL HUB MODAL LISTENERS ---
if (elements.btnOpenHub && elements.featureHubModal) {
    elements.btnOpenHub.addEventListener('click', () => {
        elements.featureHubModal.classList.remove('hidden');
    });
}

if (elements.btnCloseHub && elements.featureHubModal) {
    elements.btnCloseHub.addEventListener('click', () => {
        elements.featureHubModal.classList.add('hidden');
    });
}

if (elements.featureHubModal) {
    elements.featureHubModal.addEventListener('click', (e) => {
        if (e.target === elements.featureHubModal) {
            elements.featureHubModal.classList.add('hidden');
        }
    });
}

// --- EVENT BINDINGS ---
elements.toggleObjects.addEventListener('change', (e) => {
    detectObjectsEnabled = e.target.checked;
    showToast(`Object detection ${detectObjectsEnabled ? 'enabled' : 'disabled'}.`);
});

elements.toggleFaces.addEventListener('change', (e) => {
    detectFacesEnabled = e.target.checked;
    showToast(`Face analysis ${detectFacesEnabled ? 'enabled' : 'disabled'}.`);
});

elements.toggleHands.addEventListener('change', (e) => {
    detectHandsEnabled = e.target.checked;
    showToast(`Cyber hand detection ${detectHandsEnabled ? 'enabled' : 'disabled'}.`);
});

elements.toggleGestures.addEventListener('change', (e) => {
    detectGesturesEnabled = e.target.checked;
    showToast(`Hand gestures ${detectGesturesEnabled ? 'enabled' : 'disabled'}.`);
});

elements.toggleSpider.addEventListener('change', (e) => {
    spiderEffectEnabled = e.target.checked;
    showToast(`Spider effect ${spiderEffectEnabled ? 'enabled' : 'disabled'}.`);
});

elements.toggleNumbers.addEventListener('change', (e) => {
    showJointNumbersEnabled = e.target.checked;
    showToast(`Joint numbers ${showJointNumbersEnabled ? 'enabled' : 'disabled'}.`);
});

if (elements.toggleDrowsiness) {
    elements.toggleDrowsiness.addEventListener('change', (e) => {
        detectDrowsinessEnabled = e.target.checked;
        if (!detectDrowsinessEnabled && elements.drowsinessAlertOverlay) {
            elements.drowsinessAlertOverlay.classList.add('hidden');
        }
        showToast(`Drowsiness monitor ${detectDrowsinessEnabled ? 'enabled' : 'disabled'}.`);
    });
}

if (elements.toggleAudioAlarm) {
    elements.toggleAudioAlarm.addEventListener('change', (e) => {
        audioAlarmEnabled = e.target.checked;
        showToast(`Audio alarm ${audioAlarmEnabled ? 'enabled' : 'disabled'}.`);
    });
}

if (elements.toggleSfx) {
    elements.toggleSfx.addEventListener('change', (e) => {
        sfxEnabled = e.target.checked;
        showToast(`Cyber SFX ${sfxEnabled ? 'enabled' : 'disabled'}.`);
    });
}

if (elements.toggleVoice) {
    elements.toggleVoice.addEventListener('change', (e) => {
        voiceAssistantEnabled = e.target.checked;
        if (voiceAssistantEnabled) {
            speakVoiceAlert('AI voice assistant online');
        }
        showToast(`Voice assistant ${voiceAssistantEnabled ? 'enabled' : 'disabled'}.`);
    });
}

elements.confidenceSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    confidenceThreshold = val / 100;
    elements.confidenceVal.textContent = `${val}%`;
});

elements.cameraSelect.addEventListener('change', (e) => {
    currentDeviceId = e.target.value;
    startCamera();
});

elements.btnPause.addEventListener('click', toggleInferencePause);
elements.btnScreenshot.addEventListener('click', captureSnapshot);

elements.btnToggleHud.addEventListener('click', () => {
    hudActive = !hudActive;
    elements.btnToggleHud.classList.toggle('active', hudActive);
    showToast(`Visual overlays ${hudActive ? 'enabled' : 'disabled'}.`);
});

elements.btnRetry.addEventListener('click', startCamera);

window.addEventListener('resize', () => {
    if (stream && elements.video.readyState >= 2) {
        const w = elements.video.videoWidth;
        const h = elements.video.videoHeight;
        elements.canvas.width = w;
        elements.canvas.height = h;
        if (elements.drawingCanvas) {
            elements.drawingCanvas.width = w;
            elements.drawingCanvas.height = h;
        }
    }
});

// --- INITIALIZE APPLICATION ---
initTheme();
initTabs();
setupCameraDevices().then(() => {
    initAIModels();
});
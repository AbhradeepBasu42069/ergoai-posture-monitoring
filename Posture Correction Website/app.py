import cv2
import numpy as np
import math
from collections import deque
from flask import Flask, render_template, Response, request, jsonify
import json
from flask_socketio import SocketIO
from ultralytics import YOLO
import threading
import time
import serial

# --- FLASK SETUP ---
app = Flask(__name__)
# threading mode is best for Windows + OpenCV streaming reliability
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- CONFIGURATION ---
MODEL_PATH = "yolov8n-pose.pt"

# Thresholds
BODY_FORWARD_GOOD, BODY_FORWARD_WARN, BODY_FORWARD_BAD = 5, 12, 15
FORWARD_HEAD_GOOD, FORWARD_HEAD_WARN, FORWARD_HEAD_BAD = 0.15, 0.25, 0.35
HEAD_TILT_GOOD, HEAD_TILT_WARN, HEAD_TILT_BAD = 8, 15, 25
SHOULDER_FORWARD_GOOD, SHOULDER_FORWARD_WARN, SHOULDER_FORWARD_BAD = 0.12, 0.20, 0.30
SHOULDER_HEIGHT_GOOD, SHOULDER_HEIGHT_WARN, SHOULDER_HEIGHT_BAD = 2, 5, 10
SPINE_TILT_GOOD, SPINE_TILT_WARN, SPINE_TILT_BAD = 5, 12, 20

# Colors
COLOR_EXCELLENT = (0, 255, 0)
COLOR_GOOD = (50, 255, 50)
COLOR_WARN = (0, 165, 255)
COLOR_BAD = (0, 0, 255)
COLOR_SKELETON = (255, 180, 100)

KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- CLASSES ---
class PersonTracker:
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.confidence_threshold = 0.6
        
    def select_main_person(self, detections):
        if not detections or len(detections) == 0: return None
        best_detection = None
        best_score = -1
        
        for detection in detections:
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                # Check if boxes exist and have length
                try:
                    if detection.boxes.xyxy.shape[0] == 0 or detection.boxes.conf.shape[0] == 0:
                        continue
                    box = detection.boxes.xyxy[0].cpu().numpy()
                    conf = detection.boxes.conf[0].cpu().numpy()
                except (IndexError, RuntimeError, AttributeError):
                    continue
                if conf < self.confidence_threshold: continue
                
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                box_center_x = (x1 + x2) / 2
                frame_center_x = 640
                distance_from_center = abs(box_center_x - frame_center_x)
                score = area / (1 + distance_from_center * 0.01) * conf
                
                if score > best_score:
                    best_score = score
                    best_detection = detection
        return best_detection

class MetricSmoother:
    def __init__(self, window_size=5):
        self.metrics = {}
        self.window_size = window_size
    
    def update(self, name, value):
        if value is None: return None
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(value)
        return float(np.mean(self.metrics[name]))

# --- SINGLETON CAMERA MANAGER ---
class CameraStream:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        # OPTIMIZATION: Match display resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Loading YOLO Model...")
        self.model = YOLO(MODEL_PATH)
        self.tracker = PersonTracker()
        self.smoother = MetricSmoother(window_size=5)
        
        self.is_running = True
        self.camera_active = True
        
        # Synchronization
        self.condition = threading.Condition()
        self.lock = threading.Lock()
        
        # Shared State
        self.current_frame = None
        self.current_jpeg = None
        self.web_data = {}
        self.last_emit_time = 0
        
        # Start Thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.is_running:
            if not self.camera_active:
                time.sleep(0.5)
                continue

            success, frame = self.camera.read()
            if not success:
                self.camera.release()
                time.sleep(1)
                self.camera = cv2.VideoCapture(0)
                continue

            # Process Frame 
            frame = cv2.flip(frame, 1)
            results = self.model.track(frame, persist=True, verbose=False, conf=0.5)
            
            temp_web_data = {
                'score': 0, 'status': "Searching...", 'issues': [],
                'metrics': {'body_lean': 0, 'head_angle': 0, 'head_tilt': 0, 'shoulder_asym': 0},
                'hardware': {'temp': 36.5, 'pressure': 0}
            }

            if results and len(results) > 0:
                main_detection = self.tracker.select_main_person(results)
                if main_detection and main_detection.keypoints:
                    kp_data = main_detection.keypoints.xy
                    if len(kp_data) > 0:
                        kp = kp_data[0].cpu().numpy()
                        if kp.shape[0] == 17:
                            # Extract Points
                            nose = get_point(kp, KP["nose"])
                            left_eye, right_eye = get_point(kp, KP["left_eye"]), get_point(kp, KP["right_eye"])
                            left_ear, right_ear = get_point(kp, KP["left_ear"]), get_point(kp, KP["right_ear"])
                            left_shoulder, right_shoulder = get_point(kp, KP["left_shoulder"]), get_point(kp, KP["right_shoulder"])
                            left_hip, right_hip = get_point(kp, KP["left_hip"]), get_point(kp, KP["right_hip"])
                            left_knee, right_knee = get_point(kp, KP["left_knee"]), get_point(kp, KP["right_knee"])
                            
                            mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
                            mid_hip = get_midpoint(left_hip, right_hip)
                            mid_knee = get_midpoint(left_knee, right_knee)
                            mid_ear = get_midpoint(left_ear, right_ear)
                            shoulder_width = calculate_distance(left_shoulder, right_shoulder)

                            # 1. RAW COMPUTATIONS
                            raw_body_lean = compute_body_forward_lean(mid_shoulder, mid_hip, mid_knee)
                            fh_ratio, fh_angle = compute_forward_head_posture(nose, mid_ear, mid_shoulder)
                            head_tilt = compute_head_tilt(left_eye, right_eye, left_ear, right_ear)
                            shoulder_slouch = compute_shoulder_forward_slouch(mid_shoulder, mid_hip, shoulder_width)
                            shoulder_asym = compute_shoulder_asymmetry(left_shoulder, right_shoulder)
                            spine_tilt = compute_spine_lateral_tilt(mid_shoulder, mid_hip)

                            # 2. SMOOTHING
                            m_body = self.smoother.update('body_lean', raw_body_lean) or 0
                            m_head_ang = self.smoother.update('fh_angle', fh_angle) or 0
                            m_tilt = self.smoother.update('head_tilt', head_tilt) or 0
                            m_slouch = self.smoother.update('shoulder_slouch', shoulder_slouch) or 0
                            m_asym = self.smoother.update('shoulder_asym', shoulder_asym) or 0
                            m_spine = self.smoother.update('spine_tilt', spine_tilt) or 0
                            m_ratio = self.smoother.update('fh_ratio', fh_ratio) or 0

                            # 3. COMPOSITE
                            composite_lean = m_body + (m_head_ang * 0.3) 
                            
                            metrics_map = {
                                'body_forward_lean': composite_lean,
                                'forward_head_ratio': m_ratio,
                                'forward_head_angle': m_head_ang,
                                'head_tilt': m_tilt,
                                'shoulder_slouch': m_slouch,
                                'shoulder_asymmetry': m_asym,
                                'spine_tilt': m_spine
                            }

                            status, color, issues, score = classify_comprehensive_posture(metrics_map)
                            draw_skeleton(frame, kp, color)
                            draw_reference_lines(frame, kp, color)

                            # Hardware Sim (inside helper to keep clean usually, but logic here)
                            import random
                            hw_temp = 36.5 + random.uniform(-0.5, 0.5)
                            hw_pressure = 45 + random.randint(0, 10)

                            temp_web_data = {
                                'score': score,
                                'status': status,
                                'issues': issues,
                                'metrics': {
                                    'body_lean': round(composite_lean, 1),
                                    'head_angle': round(m_head_ang, 1),
                                    'head_tilt': round(m_tilt, 1),
                                    'shoulder_asym': round(m_asym, 1)
                                },
                                'hardware': {
                                    'temp': round(hw_temp, 1),
                                    'pressure': hw_pressure
                                }
                            }

            # Encode
            small_frame = cv2.resize(frame, (854, 480)) 
            ret, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            # Notify Waiters
            with self.condition:
                self.current_frame = frame
                self.current_jpeg = buffer.tobytes()
                self.web_data = temp_web_data
                self.condition.notify_all()
            
            # Rate Limit Socket (Max 10 per sec)
            curr_time = time.time()
            if curr_time - self.last_emit_time > 0.1:
                socketio.emit('stats_update', temp_web_data)
                self.last_emit_time = curr_time

    def get_jpeg(self):
        with self.condition:
            self.condition.wait() # Block until new frame
            return self.current_jpeg

    def toggle_camera(self, state):
        self.camera_active = state

# --- HELPER FUNCTIONS (UNCHANGED) ---
def get_point(kp, idx):
    point = kp[idx]
    if np.all(point > 0): return point
    return None

def get_midpoint(p1, p2):
    if p1 is None or p2 is None: return None
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

def calculate_distance(p1, p2):
    if p1 is None or p2 is None: return None
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def compute_body_forward_lean(mid_shoulder, mid_hip, mid_knee):
    if mid_shoulder is None or mid_hip is None: return None
    base_point = mid_knee if mid_knee is not None else mid_hip
    vertical_ref_x = mid_hip[0]
    shoulder_forward_dist = mid_shoulder[0] - vertical_ref_x
    vertical_dist = abs(mid_hip[1] - mid_shoulder[1])
    if vertical_dist < 30: return 0
    angle = math.degrees(math.atan2(abs(shoulder_forward_dist), vertical_dist))
    return angle

def compute_forward_head_posture(nose, mid_ear, mid_shoulder):
    if nose is None or mid_shoulder is None: return None, None
    reference_point = mid_ear if mid_ear is not None else nose
    horizontal_dist = abs(reference_point[0] - mid_shoulder[0])
    vertical_dist = abs(reference_point[1] - mid_shoulder[1])
    if vertical_dist < 20: return None, None
    ratio = horizontal_dist / vertical_dist
    angle = math.degrees(math.atan2(horizontal_dist, vertical_dist))
    return ratio, angle

def compute_head_tilt(left_eye, right_eye, left_ear, right_ear):
    if left_eye is not None and right_eye is not None: p1, p2 = left_eye, right_eye
    elif left_ear is not None and right_ear is not None: p1, p2 = left_ear, right_ear
    else: return None
    dy = abs(p2[1] - p1[1])
    dx = abs(p2[0] - p1[0])
    if dx < 10: return 0
    return math.degrees(math.atan2(dy, dx))

def compute_shoulder_forward_slouch(shoulders_mid, hips_mid, shoulder_width):
    if shoulders_mid is None or hips_mid is None or shoulder_width is None: return None
    horizontal_diff = abs(shoulders_mid[0] - hips_mid[0])
    if shoulder_width < 20: return None
    return horizontal_diff / shoulder_width

def compute_shoulder_asymmetry(left_shoulder, right_shoulder):
    if left_shoulder is None or right_shoulder is None: return None
    return abs(left_shoulder[1] - right_shoulder[1])

def compute_spine_lateral_tilt(shoulders_mid, hips_mid):
    if shoulders_mid is None or hips_mid is None: return None
    dx = abs(shoulders_mid[0] - hips_mid[0])
    dy = abs(shoulders_mid[1] - hips_mid[1])
    if dy < 30: return 0
    return math.degrees(math.atan2(dx, dy))

def draw_skeleton(frame, kp, color=COLOR_SKELETON):
    for i, j in SKELETON:
        p1 = get_point(kp, i)
        p2 = get_point(kp, j)
        if p1 is not None and p2 is not None:
            thickness = 4 if i in [5, 6, 11, 12] or j in [5, 6, 11, 12] else 3
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness, cv2.LINE_AA)
    for idx in range(len(kp)):
        point = get_point(kp, idx)
        if point is not None:
            cv2.circle(frame, tuple(point.astype(int)), 5, color, -1)

def draw_reference_lines(frame, kp, color=(255, 0, 255)):
    nose = get_point(kp, KP["nose"])
    left_shoulder = get_point(kp, KP["left_shoulder"])
    right_shoulder = get_point(kp, KP["right_shoulder"])
    left_hip = get_point(kp, KP["left_hip"])
    mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
    mid_hip = get_midpoint(left_hip, get_point(kp, KP["right_hip"]))
    
    if mid_hip is not None and mid_shoulder is not None:
        line_bottom = (int(mid_hip[0]), int(mid_hip[1]))
        line_top = (int(mid_hip[0]), int(mid_shoulder[1] - 100))
        cv2.line(frame, line_bottom, line_top, (200, 200, 200), 2, cv2.LINE_AA)

def classify_comprehensive_posture(metrics):
    score = 100.0
    issues = []
    
    lean = metrics.get('body_forward_lean', 0)
    if lean > BODY_FORWARD_GOOD:
        penalty = (lean - BODY_FORWARD_GOOD) * 2.5
        score -= penalty
        if lean > BODY_FORWARD_WARN:
            issues.append(f"Body Lean ({int(lean)}°)")

    head_ang = metrics.get('forward_head_angle', 0)
    if head_ang > 15:
        penalty = (head_ang - 15) * 2
        score -= penalty
        if head_ang > 25:
             issues.append("Forward Head")
    
    tilt = metrics.get('head_tilt', 0)
    if tilt > HEAD_TILT_GOOD:
        penalty = (tilt - HEAD_TILT_GOOD) * 1.5
        score -= penalty
        if tilt > HEAD_TILT_WARN:
             issues.append("Head Tilt")

    asym = metrics.get('shoulder_asymmetry', 0)
    if asym > SHOULDER_HEIGHT_GOOD:
        penalty = (asym - SHOULDER_HEIGHT_GOOD) * 0.5
        score -= penalty
        if asym > SHOULDER_HEIGHT_WARN:
             issues.append("Shoulder Uneven")

    score = max(0, min(100, score))
    
    if score >= 90:
        return "EXCELLENT", COLOR_EXCELLENT, issues, int(score)
    elif score >= 75:
        return "GOOD", COLOR_GOOD, issues, int(score)
    elif score >= 50:
        return "FAIR", COLOR_WARN, issues, int(score)
    else:
        return "POOR", COLOR_BAD, issues, int(score)

# --- GLOBAL HARDWARE STATE ---
hardware_state = {
    'pitch': 0.0,
    'slouch': 0.0,
    'alert': 0,
    'connected': False,
    'raw_log': []
}

def read_arduino_data():
    global hardware_state
    import json
    import time
    
    ser = None
    
    while True:
        # 1. Try to Connect if not connected
        if not hardware_state['connected']:
            try:
                # Try specific port first, then maybe others if needed (sticking to COM3 for now as requested/default)
                ser = serial.Serial('COM3', 9600, timeout=1) 
                hardware_state['connected'] = True
                print("Hardware Connected on COM3")
                # Clear log on new connection to avoid confusion
                hardware_state['raw_log'].append("--- CONNECTED ---")
            except Exception as e:
                # Not found, stay in simulation/disconnected mode for a bit
                hardware_state['connected'] = False
                ser = None
        
        # 2. Read Data if connected
        if hardware_state['connected'] and ser:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        # Keep a raw log
                        hardware_state['raw_log'].append(line)
                        if len(hardware_state['raw_log']) > 50: hardware_state['raw_log'].pop(0)

                        try:
                            data = json.loads(line)
                            hardware_state['pitch'] = float(data.get('pitch', 0.0))
                            hardware_state['slouch'] = float(data.get('slouch', 0.0))
                            hardware_state['alert'] = int(data.get('alert', 0))
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"Serial Error: {e}")
                hardware_state['connected'] = False
                if ser:
                    ser.close()
                ser = None
                hardware_state['raw_log'].append("--- DISCONNECTED ---")
                
        # 3. Simulation Fallback (only if not connected)
        else:
            time.sleep(0.2) # Check for reconnection every 200ms
            
            # Simulate slight fluctuations for UI testing
            import random
            current_pitch = hardware_state.get('pitch', -30.0)
            current_slouch = hardware_state.get('slouch', 20.0)
            
            hardware_state['pitch'] = round(current_pitch + random.uniform(-0.5, 0.5), 2)
            # Constrain to reasonable bounds
            if hardware_state['pitch'] < -50: hardware_state['pitch'] = -50
            if hardware_state['pitch'] > 50: hardware_state['pitch'] = 50

            hardware_state['slouch'] = round(current_slouch + random.uniform(-0.5, 0.5), 2)
            if hardware_state['slouch'] < -10: hardware_state['slouch'] = -10
            if hardware_state['slouch'] > 30: hardware_state['slouch'] = 30
            
            # Simulate Alert for testing UI
            hardware_state['alert'] = 1 if hardware_state['slouch'] > 25 else 0
            
            # Add simulation log periodically so it doesn't flood 
            if random.random() < 0.1: 
                log_line = json.dumps({
                    "pitch": hardware_state['pitch'],
                    "slouch": hardware_state['slouch'],
                    "alert": hardware_state['alert'],
                    "simulated": True
                })
                hardware_state['raw_log'].append(log_line)
                if len(hardware_state['raw_log']) > 50: hardware_state['raw_log'].pop(0)

# Start Background Threads
if not any(t.name == 'ArduinoThread' for t in threading.enumerate()):
    t = threading.Thread(target=read_arduino_data, name='ArduinoThread')
    t.daemon = True
    t.start()

# --- INITIALIZE CAMERA ---
# Initialize globally once
global_camera_stream = CameraStream()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hardware')
def hardware():
    return render_template('hardware.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/suggestions')
def suggestions():
    return render_template('suggestions.html')

@app.route('/break')
def take_break():
    return render_template('break.html')

@app.route('/api/hardware_data')
def api_hardware_data():
    return jsonify(hardware_state)


@app.route('/api/hardware_update', methods=['POST'])
def api_hardware_update():
    """Endpoint for Arduino (or any device) to POST latest hardware telemetry.
    Accepts JSON like: {"pitch": -12.3, "slouch": 18.2, "alert": 0, "connected": true}
    """
    global hardware_state
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'invalid json'}), 400

    if not data:
        return jsonify({'status': 'error', 'message': 'no data provided'}), 400

    # Safely update known fields
    try:
        if 'pitch' in data:
            hardware_state['pitch'] = float(data.get('pitch', hardware_state.get('pitch', 0.0)))
        if 'slouch' in data:
            hardware_state['slouch'] = float(data.get('slouch', hardware_state.get('slouch', 0.0)))
        if 'alert' in data:
            hardware_state['alert'] = int(data.get('alert', hardware_state.get('alert', 0)))
        if 'connected' in data:
            hardware_state['connected'] = bool(data.get('connected'))

        # Append raw incoming JSON to raw_log for display (keep bounded)
        hardware_state['raw_log'].append(json.dumps(data))
        if len(hardware_state['raw_log']) > 50:
            hardware_state['raw_log'].pop(0)

        # Optionally emit a realtime update to any websocket listeners
        try:
            # Emit hardware state including a trimmed raw_log for realtime clients
            socketio.emit('hardware_update', {
                'pitch': hardware_state['pitch'],
                'slouch': hardware_state['slouch'],
                'alert': hardware_state['alert'],
                'connected': hardware_state['connected'],
                'raw_log': hardware_state.get('raw_log', [])[-50:]
            })
        except Exception:
            pass

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

    return jsonify({'status': 'ok'})

def stream_gen():
    while True:
        # Blocks until new frame matches
        frame_bytes = global_camera_stream.get_jpeg()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # No else/sleep needed as get_jpeg waits with Condition

@app.route('/video_feed')
def video_feed():
    return Response(stream_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('toggle_camera')
def handle_camera_toggle(message):
    active = message.get('active', True)
    global_camera_stream.toggle_camera(active)
    print(f"Camera Toggled: {active}")

if __name__ == '__main__':
    # Fix for WinError 10038: Add use_reloader=False to prevent duplicate initialization
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True, use_reloader=False)
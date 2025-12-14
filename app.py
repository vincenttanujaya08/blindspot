"""
Flask Server for ADAS Blind Spot Detection System
Real-time video streaming with detection overlay
"""

from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import cv2
import threading
import time
import os
from logic.unified_detector import UnifiedBlindSpotDetector

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_SOURCES = {
    'front': 'camera/front.mp4',
    'back': 'camera/back.mp4',
    'left': 'camera/left.mp4',
    'right': 'camera/right.mp4'
}

MODEL_PATH = 'best_yolov11n_BDD100K_50.pt'

# Global detector instance
detector = None
detector_lock = threading.Lock()

# Camera states
camera_states = {
    'front': {'level': 0, 'threat': 'SAFE', 'detections': []},
    'back': {'level': 0, 'threat': 'SAFE', 'detections': []},
    'left': {'level': 0, 'threat': 'SAFE', 'detections': []},
    'right': {'level': 0, 'threat': 'SAFE', 'detections': []}
}

# System state
system_running = False

# ============================================================
# DETECTOR INITIALIZATION
# ============================================================

def init_detector():
    """Initialize the detector (lazy loading)"""
    global detector
    if detector is None:
        with detector_lock:
            if detector is None:  # Double-check
                print("🚗 Initializing detector...")
                detector = UnifiedBlindSpotDetector(
                    model_path=MODEL_PATH,
                    use_gpu=True
                )
                print("✓ Detector ready\n")
    return detector

# ============================================================
# VIDEO FRAME GENERATOR
# ============================================================

def generate_frames(camera_id):
    """
    Generator function for video streaming
    Yields JPEG frames with detection overlay
    """
    global system_running
    
    video_path = VIDEO_SOURCES.get(camera_id)
    
    if not video_path or not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        # Generate placeholder frame
        placeholder = create_placeholder_frame(camera_id)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 fps
        return
    
    # Initialize detector
    det = init_detector()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    print(f"📹 Streaming {camera_id} camera...")
    frame_count = 0

    try:
        while True:
            if not system_running:
                # System stopped - show static frame
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                
                # Just encode without processing
                frame = cv2.resize(frame, (640, 360))
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)
                continue
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_count += 1

            if frame_count % 3 != 0:
                time.sleep(0.01)
                continue
            # Process frame with detection
            result = det.process_frame(frame, camera_id)
            if result is None or len(result) != 3:
                continue
            processed_frame, detections, threat_level = result
            
            # Update camera state
            level = 0
            if threat_level == "WARNING":
                level = 1
            elif threat_level == "DANGER":
                level = 2
            
            camera_states[camera_id] = {
                'level': level,
                'threat': threat_level,
                'detections': len(detections)
            }
            processed_frame = cv2.resize(processed_frame, (640, 360))

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except GeneratorExit:
        print(f"🛑 Stopped streaming {camera_id}")
    finally:
        cap.release()

def create_placeholder_frame(camera_id):
    """Create placeholder frame when video not found"""
    frame = cv2.imread('static/placeholder.jpg') if os.path.exists('static/placeholder.jpg') else None
    
    if frame is None:
        # Create simple colored placeholder
        colors = {
            'front': (100, 50, 50),
            'back': (50, 100, 50),
            'left': (50, 50, 100),
            'right': (100, 100, 50)
        }
        frame = (colors.get(camera_id, (50, 50, 50)) * np.ones((480, 640, 3))).astype(np.uint8)
    
    cv2.putText(frame, f"{camera_id.upper()} CAMERA", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, "Video Not Found", (230, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return frame

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve main HTML page"""
    return render_template('index.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Video streaming route"""
    if camera_id not in VIDEO_SOURCES:
        return "Invalid camera ID", 404
    
    return Response(
        generate_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/start', methods=['POST'])
def start_engine():
    """Start the detection system"""
    global system_running
    system_running = True
    
    return jsonify({
        'status': 'success',
        'message': 'Engine started',
        'running': system_running
    })

@app.route('/api/stop', methods=['POST'])
def stop_engine():
    """Stop the detection system"""
    global system_running
    system_running = False
    
    # Reset camera states
    for cam in camera_states:
        camera_states[cam] = {'level': 0, 'threat': 'SAFE', 'detections': []}
    
    return jsonify({
        'status': 'success',
        'message': 'Engine stopped',
        'running': system_running
    })

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'running': system_running,
        'cameras': camera_states
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'video_sources': {
            cam: os.path.exists(path) 
            for cam, path in VIDEO_SOURCES.items()
        }
    })

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("🚗 ADAS Blind Spot Detection System")
    print("="*60)
    
    # Check video files
    print("\n📹 Checking video sources...")
    all_found = True
    for cam, path in VIDEO_SOURCES.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "❌"
        print(f"  {status} {cam}: {path}")
        if not exists:
            all_found = False
    
    if not all_found:
        print("\n⚠ WARNING: Some video files not found!")
        print("  Placeholder frames will be used instead.")
    
    # Check model
    print(f"\n🤖 Checking model...")
    if os.path.exists(MODEL_PATH):
        print(f"  ✓ Model found: {MODEL_PATH}")
    else:
        print(f"  ❌ Model not found: {MODEL_PATH}")
        print("  Please download the model file!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🌐 Starting Flask server...")
    print("="*60)
    print(f"\n  📱 Open browser: http://localhost:5000")
    print(f"  🛑 Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        threaded=True
    )
"""
Unified Blind Spot Detector for ADAS System
Combines front, back, left, and right camera detection logic
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class UnifiedBlindSpotDetector:
    """
    Unified detector that handles all 4 camera angles with their specific logic
    """
    """
Unified Blind Spot Detector for ADAS System
Combines front, back, left, and right camera detection logic
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class UnifiedBlindSpotDetector:
    """
    Unified detector that handles all 4 camera angles with their specific logic
    """
    
    def __init__(self, model_path='best_yolov11n_BDD100K_50.pt', use_gpu=True):
        print("🚗 Initializing ADAS Blind Spot Detection System...")
    
    # Load YOLO model
        try:
            self.model = YOLO(model_path)
            
            # PATCH 1: Disable model.fuse
            self.model.fuse = lambda *args, **kwargs: self.model
            
            # PATCH 2: Patch the actual model's fuse method
            if hasattr(self.model.model, 'fuse'):
                self.model.model.fuse = lambda *args, **kwargs: self.model.model
            
            # PATCH 3: Monkey patch tasks.py fuse function
            from ultralytics.nn import tasks
            _original_fuse = tasks.DetectionModel.fuse
            def safe_fuse(self, *args, **kwargs):
                try:
                    return _original_fuse(self, *args, **kwargs)
                except AttributeError as e:
                    if 'bn' in str(e):
                        print("⚠ Skipping model fusion due to compatibility issue")
                        return self
                    raise
            tasks.DetectionModel.fuse = safe_fuse
            
            print(f"✓ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # GPU setup
        if use_gpu:
            try:
                self.model.to('cuda')
                print("✓ Using GPU (CUDA)")
            except:
                print("⚠ GPU unavailable, using CPU")
        else:
            print("✓ Using CPU")
        
        # Detection thresholds per camera type
        self.thresholds = {
            'front': {'danger': 0.4, 'warning': 0.25},
            'back': {'danger': 0.4, 'warning': 0.25},
            'left': {'danger': 0.05, 'warning': 0.03},
            'right': {'danger': 0.5, 'warning': 0.30}
        }
        
        # State tracking
        self.prev_gray = {}  # camera_id -> grayscale frame
        self.flow_memory = {}  # camera_id -> track_id -> flow data
        self.temporal_memory = {}  # camera_id -> track_id -> temporal data
        self.fps_history = {}  # camera_id -> deque of fps values
        self.prev_time = {}  # camera_id -> timestamp
        self.detection_zones = {}  # camera_id -> polygon
        
        # Config
        self.temporal_alpha = 0.6  # EMA smoothing factor
        self.debug = False
        
        print("✓ Detector initialized successfully\n")
        
    # ============================================================
    # ZONE CREATION (Camera-specific)
    # ============================================================
    
    def create_zone_front(self, w, h):
        """Front camera: Focus on upper-middle area"""
        return np.array([
            [int(w*0.15), int(h*0.8)],
            [int(w*0.15), int(h*0.5)],
            [int(w*0.8), int(h*0.5)],
            [int(w*0.8), int(h*0.8)]
        ], np.int32)
    
    def create_zone_back(self, w, h):
        """Back camera: Focus on lower area (rear approach)"""
        return np.array([
            [int(w*0.25), int(h*1.0)],
            [int(w*0.25), int(h*0.8)],
            [int(w*0.75), int(h*0.8)],
            [int(w*0.75), int(h*1.0)]
        ], np.int32)
    
    def create_zone_left(self, w, h):
        """Left camera: Diagonal blind spot zone"""
        return np.array([
            [int(w*0.05), int(h*0.85)],
            [int(w*0.25), int(h*0.75)],
            [int(w*0.75), int(h*0.95)],
            [int(w*0.65), int(h*1.25)]
        ], np.int32)
    
    def create_zone_right(self, w, h):
        """Right camera: Angled blind spot zone (camera pointing ~30° forward)"""
        return np.array([
            [int(w*0.75), int(h*1.3)],
            [int(w*0.5), int(h*1.1)],
            [int(w*0.80), int(h*0.65)],
            [int(w*1.00), int(h*0.7)]
        ], np.int32)
    
    def get_detection_zone(self, camera_id, w, h):
        """Get or create detection zone for camera"""
        if camera_id not in self.detection_zones:
            if camera_id == 'front':
                self.detection_zones[camera_id] = self.create_zone_front(w, h)
            elif camera_id == 'back':
                self.detection_zones[camera_id] = self.create_zone_back(w, h)
            elif camera_id == 'left':
                self.detection_zones[camera_id] = self.create_zone_left(w, h)
            elif camera_id == 'right':
                self.detection_zones[camera_id] = self.create_zone_right(w, h)
        
        return self.detection_zones[camera_id]
    
    # ============================================================
    # GEOMETRIC CALCULATIONS
    # ============================================================
    
    def is_horizontally_in_zone(self, bbox, polygon):
        """Check if bbox center-x is within zone horizontal bounds"""
        x, y, w, h = bbox
        bbox_cx = x + w * 0.5
        
        zone_x = polygon[:, 0]
        zone_left = np.min(zone_x)
        zone_right = np.max(zone_x)
        
        return zone_left <= bbox_cx <= zone_right
    
    def bbox_zone_vertical_overlap(self, bbox, polygon, frame_shape):
        """Calculate vertical (Y-axis) overlap ratio"""
        _, h_img = frame_shape[:2]
        
        x, y, w, h = bbox
        bbox_top = y
        bbox_bottom = y + h
        
        zone_y = polygon[:, 1]
        zone_top = np.min(zone_y)
        zone_bottom = np.max(zone_y)
        
        overlap_top = max(bbox_top, zone_top)
        overlap_bottom = min(bbox_bottom, zone_bottom)
        
        overlap_h = max(0, overlap_bottom - overlap_top)
        zone_h = zone_bottom - zone_top
        
        return overlap_h / zone_h if zone_h > 0 else 0.0
    
    def bbox_zone_overlap_ratio(self, bbox, polygon, frame_shape):
        """Calculate full area overlap ratio (for left/right cameras)"""
        h_img, w_img = frame_shape[:2]
        
        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        zone_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        x, y, w, h = bbox
        cv2.rectangle(bbox_mask, (x, y), (x + w, y + h), 255, -1)
        cv2.fillPoly(zone_mask, [polygon], 255)
        
        intersection = cv2.bitwise_and(bbox_mask, zone_mask)
        inter_area = cv2.countNonZero(intersection)
        zone_area = cv2.countNonZero(zone_mask)
        
        return inter_area / zone_area if zone_area > 0 else 0.0
    
    # ============================================================
    # OPTICAL FLOW (for Right camera)
    # ============================================================
    
    def compute_lateral_flow(self, camera_id, prev_gray, curr_gray, bbox, track_id):
        """Compute optical flow for lateral movement detection"""
        x, y, w, h = bbox
        h_img, w_img = curr_gray.shape[:2]
        
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        
        prev_roi = prev_gray[y:y+h, x:x+w]
        curr_roi = curr_gray[y:y+h, x:x+w]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        
        # EMA smoothing per track
        flow_key = f"{camera_id}_{track_id}"
        if flow_key not in self.flow_memory:
            self.flow_memory[flow_key] = (dx, dy)
        else:
            px, py = self.flow_memory[flow_key]
            dx = 0.7 * px + 0.3 * dx
            dy = 0.7 * py + 0.3 * dy
            self.flow_memory[flow_key] = (dx, dy)
        
        return dx, dy
    
    # ============================================================
    # TEMPORAL DEPTH SCORE (for Left camera)
    # ============================================================
    
    def temporal_depth_score(self, camera_id, track_id, bbox, frame_height):
        """Calculate temporal depth score based on size changes"""
        x, y, w, h = bbox
        area = w * h
        bottom = y + h
        now = time.time()
        
        temp_key = f"{camera_id}_{track_id}"
        
        if temp_key not in self.temporal_memory:
            self.temporal_memory[temp_key] = {
                "h": h,
                "area": area,
                "bottom": bottom,
                "time": now,
                "ema": 0.0
            }
            return 0.0
        
        prev = self.temporal_memory[temp_key]
        dt = now - prev["time"]
        
        if dt <= 0:
            return prev["ema"]
        
        dh = max(0, (h - prev["h"]) / frame_height)
        darea = max(0, (area - prev["area"]) / (frame_height ** 2))
        dbottom = max(0, (bottom - prev["bottom"]) / frame_height)
        
        raw_score = (dh * 0.5) + (darea * 0.3) + (dbottom * 0.2)
        
        # EMA smoothing
        ema = self.temporal_alpha * raw_score + (1 - self.temporal_alpha) * prev["ema"]
        
        self.temporal_memory[temp_key] = {
            "h": h,
            "area": area,
            "bottom": bottom,
            "time": now,
            "ema": ema
        }
        
        return min(ema, 1.0)
    
    # ============================================================
    # PROXIMITY CALCULATION (Camera-specific logic)
    # ============================================================
    
    def calculate_proximity_front_back(self, camera_id, bbox, frame_shape):
        """Front/Back camera: Vertical overlap based"""
        zone = self.detection_zones[camera_id]
        
        # Horizontal gating
        if not self.is_horizontally_in_zone(bbox, zone):
            return "SAFE", (0, 255, 0), 0.0, {
                "vertical_overlap": 0.0,
                "score": 0.0,
                "gated": True
            }
        
        vertical_overlap = self.bbox_zone_vertical_overlap(bbox, zone, frame_shape)
        score = vertical_overlap ** 1.5  # Emphasize near-bottom danger
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger']:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning']:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "vertical_overlap": vertical_overlap,
            "score": score,
            "gated": False
        }
        
        return level, color, score, debug
    
    def calculate_proximity_left(self, camera_id, bbox, frame_shape, track_id):
        """Left camera: Area overlap + temporal depth"""
        zone = self.detection_zones[camera_id]
        
        overlap = self.bbox_zone_overlap_ratio(bbox, zone, frame_shape)
        temporal = self.temporal_depth_score(camera_id, track_id, bbox, frame_shape[0])
        
        # Fused score
        score = overlap * 0.6 + temporal * 0.4
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger']:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning']:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "overlap": overlap,
            "temporal": temporal,
            "score": score
        }
        
        return level, color, score, debug
    
    def calculate_proximity_right(self, camera_id, bbox, frame_shape, dx, dy):
        """Right camera: Area overlap + optical flow"""
        zone = self.detection_zones[camera_id]
        
        overlap = self.bbox_zone_overlap_ratio(bbox, zone, frame_shape)
        
        # Blind-spot direction vector
        blindspot_dir = np.array([0.4, 1])
        blindspot_dir /= np.linalg.norm(blindspot_dir)
        
        motion_vec = np.array([dx, dy])
        proj = np.dot(motion_vec, blindspot_dir)
        
        # Only care about motion INTO blind spot
        motion_score = max(0.0, proj)
        motion_score = np.clip(motion_score / 5.0, 0.0, 1.0)
        
        score = 0.6 * overlap + 0.4 * motion_score
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger'] and overlap > 0.6:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning'] and overlap > 0.3:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "overlap": overlap,
            "dx": dx,
            "dy": dy,
            "motion": motion_score,
            "score": score
        }
        
        return level, color, score, debug
    
    # ============================================================
    # FPS CALCULATION
    # ============================================================
    
    def calculate_fps(self, camera_id):
        """Calculate FPS for specific camera"""
        if camera_id not in self.fps_history:
            self.fps_history[camera_id] = deque(maxlen=20)
            self.prev_time[camera_id] = time.time()
        
        now = time.time()
        time_diff = now - self.prev_time[camera_id]
        fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time[camera_id] = now
        self.fps_history[camera_id].append(fps)
        
        return sum(self.fps_history[camera_id]) / len(self.fps_history[camera_id])
    
    # ============================================================
    # DRAWING UTILITIES
    # ============================================================
    
    def draw_zone(self, frame, zone):
        """Draw detection zone overlay"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 140, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [zone], True, (0, 140, 255), 2)
    
    def draw_debug_info(self, frame, bbox, debug_data):
        """Draw debug information"""
        x, y, _, _ = bbox
        
        lines = []
        for key, value in debug_data.items():
            if key != 'gated':
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.2f}")
                else:
                    lines.append(f"{key}: {value}")
        
        for i, text in enumerate(lines):
            cv2.putText(
                frame, text,
                (x, y - 10 - i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1
            )
    
    # ============================================================
    # MAIN PROCESSING
    # ============================================================
    
    def process_frame(self, frame, camera_id='front'):
        """
        Process a single frame from specified camera
        
        Args:
            frame: Input frame (BGR)
            camera_id: One of 'front', 'back', 'left', 'right'
        
        Returns:
            processed_frame: Annotated frame
            detections: List of detection results
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get detection zone
        zone = self.get_detection_zone(camera_id, w, h)
        
        # Run YOLO tracking
        results = self.model.track(
            frame,
            conf=0.5,
            iou=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            
        )
        
        # Initialize prev_gray for this camera
        if camera_id not in self.prev_gray:
            self.prev_gray[camera_id] = gray
            return frame, []
        
        detections = []
        max_threat_level = "SAFE"
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            for box in r.boxes:
                if box.id is None:
                    continue
                
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Calculate proximity based on camera type
                if camera_id in ['front', 'back']:
                    level, color, score, debug_data = self.calculate_proximity_front_back(
                        camera_id, bbox, frame.shape
                    )
                
                elif camera_id == 'left':
                    level, color, score, debug_data = self.calculate_proximity_left(
                        camera_id, bbox, frame.shape, track_id
                    )
                
                elif camera_id == 'right':
                    dx, dy = self.compute_lateral_flow(
                        camera_id, self.prev_gray[camera_id], gray, bbox, track_id
                    )
                    level, color, score, debug_data = self.calculate_proximity_right(
                        camera_id, bbox, frame.shape, dx, dy
                    )
                
                # Track max threat
                if level == "DANGER":
                    max_threat_level = "DANGER"
                elif level == "WARNING" and max_threat_level != "DANGER":
                    max_threat_level = "WARNING"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{level}"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )
                
                # Debug overlay
                if self.debug:
                    self.draw_debug_info(frame, bbox, debug_data)
                
                # Store detection
                detections.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'class': cls,
                    'confidence': conf,
                    'level': level,
                    'score': score,
                    'debug': debug_data
                })
        
        # Draw zone
        self.draw_zone(frame, zone)
        
        # Draw FPS
        fps = self.calculate_fps(camera_id)
        cv2.putText(
            frame, f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
        
        # Draw camera label
        cv2.putText(
            frame, f"{camera_id.upper()} CAM",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        
        # Update prev_gray
        self.prev_gray[camera_id] = gray
        
        return frame, detections, max_threat_level


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Test detector
    detector = UnifiedBlindSpotDetector(
        model_path='best_yolov11n_BDD100K_50.pt',
        use_gpu=True
    )
    
    # Test with video if path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        camera_id = sys.argv[2] if len(sys.argv) > 2 else 'front'
        
        print(f"\n🎥 Testing with video: {video_path}")
        print(f"📹 Camera: {camera_id}\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            sys.exit(1)
        
        cv2.namedWindow("ADAS Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ADAS Test", 960, 540)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed, detections, threat = detector.process_frame(frame, camera_id)
            
            cv2.imshow("ADAS Test", processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("✓ Detector initialized successfully")
        print("\nUsage: python unified_detector.py <video_path> <camera_id>")
        print("Example: python unified_detector.py camera/front.mp4 front")
    
        
    # ============================================================
    # ZONE CREATION (Camera-specific)
    # ============================================================
    
    def create_zone_front(self, w, h):
        """Front camera: Focus on upper-middle area"""
        return np.array([
            [int(w*0.15), int(h*0.8)],
            [int(w*0.15), int(h*0.5)],
            [int(w*0.8), int(h*0.5)],
            [int(w*0.8), int(h*0.8)]
        ], np.int32)
    
    def create_zone_back(self, w, h):
        """Back camera: Focus on lower area (rear approach)"""
        return np.array([
            [int(w*0.25), int(h*1.0)],
            [int(w*0.25), int(h*0.8)],
            [int(w*0.75), int(h*0.8)],
            [int(w*0.75), int(h*1.0)]
        ], np.int32)
    
    def create_zone_left(self, w, h):
        """Left camera: Diagonal blind spot zone"""
        return np.array([
            [int(w*0.05), int(h*0.85)],
            [int(w*0.25), int(h*0.75)],
            [int(w*0.75), int(h*0.95)],
            [int(w*0.65), int(h*1.25)]
        ], np.int32)
    
    def create_zone_right(self, w, h):
        """Right camera: Angled blind spot zone (camera pointing ~30° forward)"""
        return np.array([
            [int(w*0.75), int(h*1.3)],
            [int(w*0.5), int(h*1.1)],
            [int(w*0.80), int(h*0.65)],
            [int(w*1.00), int(h*0.7)]
        ], np.int32)
    
    def get_detection_zone(self, camera_id, w, h):
        """Get or create detection zone for camera"""
        if camera_id not in self.detection_zones:
            if camera_id == 'front':
                self.detection_zones[camera_id] = self.create_zone_front(w, h)
            elif camera_id == 'back':
                self.detection_zones[camera_id] = self.create_zone_back(w, h)
            elif camera_id == 'left':
                self.detection_zones[camera_id] = self.create_zone_left(w, h)
            elif camera_id == 'right':
                self.detection_zones[camera_id] = self.create_zone_right(w, h)
        
        return self.detection_zones[camera_id]
    
    # ============================================================
    # GEOMETRIC CALCULATIONS
    # ============================================================
    
    def is_horizontally_in_zone(self, bbox, polygon):
        """Check if bbox center-x is within zone horizontal bounds"""
        x, y, w, h = bbox
        bbox_cx = x + w * 0.5
        
        zone_x = polygon[:, 0]
        zone_left = np.min(zone_x)
        zone_right = np.max(zone_x)
        
        return zone_left <= bbox_cx <= zone_right
    
    def bbox_zone_vertical_overlap(self, bbox, polygon, frame_shape):
        """Calculate vertical (Y-axis) overlap ratio"""
        _, h_img = frame_shape[:2]
        
        x, y, w, h = bbox
        bbox_top = y
        bbox_bottom = y + h
        
        zone_y = polygon[:, 1]
        zone_top = np.min(zone_y)
        zone_bottom = np.max(zone_y)
        
        overlap_top = max(bbox_top, zone_top)
        overlap_bottom = min(bbox_bottom, zone_bottom)
        
        overlap_h = max(0, overlap_bottom - overlap_top)
        zone_h = zone_bottom - zone_top
        
        return overlap_h / zone_h if zone_h > 0 else 0.0
    
    def bbox_zone_overlap_ratio(self, bbox, polygon, frame_shape):
        """Calculate full area overlap ratio (for left/right cameras)"""
        h_img, w_img = frame_shape[:2]
        
        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        zone_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        x, y, w, h = bbox
        cv2.rectangle(bbox_mask, (x, y), (x + w, y + h), 255, -1)
        cv2.fillPoly(zone_mask, [polygon], 255)
        
        intersection = cv2.bitwise_and(bbox_mask, zone_mask)
        inter_area = cv2.countNonZero(intersection)
        zone_area = cv2.countNonZero(zone_mask)
        
        return inter_area / zone_area if zone_area > 0 else 0.0
    
    # ============================================================
    # OPTICAL FLOW (for Right camera)
    # ============================================================
    
    def compute_lateral_flow(self, camera_id, prev_gray, curr_gray, bbox, track_id):
        """Compute optical flow for lateral movement detection"""
        x, y, w, h = bbox
        h_img, w_img = curr_gray.shape[:2]
        
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        
        prev_roi = prev_gray[y:y+h, x:x+w]
        curr_roi = curr_gray[y:y+h, x:x+w]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        
        # EMA smoothing per track
        flow_key = f"{camera_id}_{track_id}"
        if flow_key not in self.flow_memory:
            self.flow_memory[flow_key] = (dx, dy)
        else:
            px, py = self.flow_memory[flow_key]
            dx = 0.7 * px + 0.3 * dx
            dy = 0.7 * py + 0.3 * dy
            self.flow_memory[flow_key] = (dx, dy)
        
        return dx, dy
    
    # ============================================================
    # TEMPORAL DEPTH SCORE (for Left camera)
    # ============================================================
    
    def temporal_depth_score(self, camera_id, track_id, bbox, frame_height):
        """Calculate temporal depth score based on size changes"""
        x, y, w, h = bbox
        area = w * h
        bottom = y + h
        now = time.time()
        
        temp_key = f"{camera_id}_{track_id}"
        
        if temp_key not in self.temporal_memory:
            self.temporal_memory[temp_key] = {
                "h": h,
                "area": area,
                "bottom": bottom,
                "time": now,
                "ema": 0.0
            }
            return 0.0
        
        prev = self.temporal_memory[temp_key]
        dt = now - prev["time"]
        
        if dt <= 0:
            return prev["ema"]
        
        dh = max(0, (h - prev["h"]) / frame_height)
        darea = max(0, (area - prev["area"]) / (frame_height ** 2))
        dbottom = max(0, (bottom - prev["bottom"]) / frame_height)
        
        raw_score = (dh * 0.5) + (darea * 0.3) + (dbottom * 0.2)
        
        # EMA smoothing
        ema = self.temporal_alpha * raw_score + (1 - self.temporal_alpha) * prev["ema"]
        
        self.temporal_memory[temp_key] = {
            "h": h,
            "area": area,
            "bottom": bottom,
            "time": now,
            "ema": ema
        }
        
        return min(ema, 1.0)
    
    # ============================================================
    # PROXIMITY CALCULATION (Camera-specific logic)
    # ============================================================
    
    def calculate_proximity_front_back(self, camera_id, bbox, frame_shape):
        """Front/Back camera: Vertical overlap based"""
        zone = self.detection_zones[camera_id]
        
        # Horizontal gating
        if not self.is_horizontally_in_zone(bbox, zone):
            return "SAFE", (0, 255, 0), 0.0, {
                "vertical_overlap": 0.0,
                "score": 0.0,
                "gated": True
            }
        
        vertical_overlap = self.bbox_zone_vertical_overlap(bbox, zone, frame_shape)
        score = vertical_overlap ** 1.5  # Emphasize near-bottom danger
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger']:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning']:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "vertical_overlap": vertical_overlap,
            "score": score,
            "gated": False
        }
        
        return level, color, score, debug
    
    def calculate_proximity_left(self, camera_id, bbox, frame_shape, track_id):
        """Left camera: Area overlap + temporal depth"""
        zone = self.detection_zones[camera_id]
        
        overlap = self.bbox_zone_overlap_ratio(bbox, zone, frame_shape)
        temporal = self.temporal_depth_score(camera_id, track_id, bbox, frame_shape[0])
        
        # Fused score
        score = overlap * 0.6 + temporal * 0.4
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger']:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning']:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "overlap": overlap,
            "temporal": temporal,
            "score": score
        }
        
        return level, color, score, debug
    
    def calculate_proximity_right(self, camera_id, bbox, frame_shape, dx, dy):
        """Right camera: Area overlap + optical flow"""
        zone = self.detection_zones[camera_id]
        
        overlap = self.bbox_zone_overlap_ratio(bbox, zone, frame_shape)
        
        # Blind-spot direction vector
        blindspot_dir = np.array([0.4, 1])
        blindspot_dir /= np.linalg.norm(blindspot_dir)
        
        motion_vec = np.array([dx, dy])
        proj = np.dot(motion_vec, blindspot_dir)
        
        # Only care about motion INTO blind spot
        motion_score = max(0.0, proj)
        motion_score = np.clip(motion_score / 5.0, 0.0, 1.0)
        
        score = 0.6 * overlap + 0.4 * motion_score
        
        thresh = self.thresholds[camera_id]
        if score > thresh['danger'] and overlap > 0.6:
            level, color = "DANGER", (0, 0, 255)
        elif score > thresh['warning'] and overlap > 0.3:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)
        
        debug = {
            "overlap": overlap,
            "dx": dx,
            "dy": dy,
            "motion": motion_score,
            "score": score
        }
        
        return level, color, score, debug
    
    # ============================================================
    # FPS CALCULATION
    # ============================================================
    
    def calculate_fps(self, camera_id):
        """Calculate FPS for specific camera"""
        if camera_id not in self.fps_history:
            self.fps_history[camera_id] = deque(maxlen=20)
            self.prev_time[camera_id] = time.time()
        
        now = time.time()
        time_diff = now - self.prev_time[camera_id]
        fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time[camera_id] = now
        self.fps_history[camera_id].append(fps)
        
        return sum(self.fps_history[camera_id]) / len(self.fps_history[camera_id])
    
    # ============================================================
    # DRAWING UTILITIES
    # ============================================================
    
    def draw_zone(self, frame, zone):
        """Draw detection zone overlay"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 140, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [zone], True, (0, 140, 255), 2)
    
    def draw_debug_info(self, frame, bbox, debug_data):
        """Draw debug information"""
        x, y, _, _ = bbox
        
        lines = []
        for key, value in debug_data.items():
            if key != 'gated':
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.2f}")
                else:
                    lines.append(f"{key}: {value}")
        
        for i, text in enumerate(lines):
            cv2.putText(
                frame, text,
                (x, y - 10 - i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1
            )
    
    # ============================================================
    # MAIN PROCESSING
    # ============================================================
    
    def process_frame(self, frame, camera_id='front'):
        """
        Process a single frame from specified camera
        
        Args:
            frame: Input frame (BGR)
            camera_id: One of 'front', 'back', 'left', 'right'
        
        Returns:
            processed_frame: Annotated frame
            detections: List of detection results
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get detection zone
        zone = self.get_detection_zone(camera_id, w, h)
        
        # Run YOLO tracking
        results = self.model.track(
            frame,
            conf=0.5,
            iou=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            
        )
        
        # Initialize prev_gray for this camera
        if camera_id not in self.prev_gray:
            self.prev_gray[camera_id] = gray
            return frame, []
        
        detections = []
        max_threat_level = "SAFE"
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            for box in r.boxes:
                if box.id is None:
                    continue
                
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Calculate proximity based on camera type
                if camera_id in ['front', 'back']:
                    level, color, score, debug_data = self.calculate_proximity_front_back(
                        camera_id, bbox, frame.shape
                    )
                
                elif camera_id == 'left':
                    level, color, score, debug_data = self.calculate_proximity_left(
                        camera_id, bbox, frame.shape, track_id
                    )
                
                elif camera_id == 'right':
                    dx, dy = self.compute_lateral_flow(
                        camera_id, self.prev_gray[camera_id], gray, bbox, track_id
                    )
                    level, color, score, debug_data = self.calculate_proximity_right(
                        camera_id, bbox, frame.shape, dx, dy
                    )
                
                # Track max threat
                if level == "DANGER":
                    max_threat_level = "DANGER"
                elif level == "WARNING" and max_threat_level != "DANGER":
                    max_threat_level = "WARNING"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{level}"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )
                
                # Debug overlay
                if self.debug:
                    self.draw_debug_info(frame, bbox, debug_data)
                
                # Store detection
                detections.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'class': cls,
                    'confidence': conf,
                    'level': level,
                    'score': score,
                    'debug': debug_data
                })
        
        # Draw zone
        self.draw_zone(frame, zone)
        
        # Draw FPS
        fps = self.calculate_fps(camera_id)
        cv2.putText(
            frame, f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
        
        # Draw camera label
        cv2.putText(
            frame, f"{camera_id.upper()} CAM",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        
        # Update prev_gray
        self.prev_gray[camera_id] = gray
        
        return frame, detections, max_threat_level


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Test detector
    detector = UnifiedBlindSpotDetector(
        model_path='best_yolov11n_BDD100K_50.pt',
        use_gpu=True
    )
    
    # Test with video if path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        camera_id = sys.argv[2] if len(sys.argv) > 2 else 'front'
        
        print(f"\n🎥 Testing with video: {video_path}")
        print(f"📹 Camera: {camera_id}\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            sys.exit(1)
        
        cv2.namedWindow("ADAS Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ADAS Test", 960, 540)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed, detections, threat = detector.process_frame(frame, camera_id)
            
            cv2.imshow("ADAS Test", processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("✓ Detector initialized successfully")
        print("\nUsage: python unified_detector.py <video_path> <camera_id>")
        print("Example: python unified_detector.py camera/front.mp4 front")
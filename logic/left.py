import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class BlindSpotDetector:
    def __init__(self, model_path='best_yolov11n_BDD100K_50.pt', use_gpu=True):
        # Load YOLO model
        self.debug = False   # set False to disable debug overlay
        self.temporal_memory = {}
        self.temporal_alpha = 0.6  # EMA smoothing factor

        self.model = YOLO(model_path)

        # Use GPU if available
        if use_gpu:
            try:
                self.model.to('cuda')
                print("✓ Using GPU")
            except:
                print("⚠ GPU unavailable, using CPU")
        else:
            print("✓ Using CPU")

        # Settings
        self.danger_threshold = 0.05
        self.warning_threshold = 0.03
        self.detected_threshold = 0.01

        self.track_history = {}
        self.warning_cooldown = {}
        self.fps_history = deque(maxlen=20)
        self.prev_time = time.time()

        self.detection_zone = None
        self.detection_width = 960

    # --------------------------------------------------------------------
    # ZONE CREATOR
    # --------------------------------------------------------------------

    def create_zone_polygons(self, width, height):
        self.detection_zone = np.array([
            [int(width*0.05), int(height*0.85)],
            [int(width*0.25), int(height*0.75)],
            [int(width*0.75), int(height*0.95)],
            [int(width*0.65), int(height*1.25)]
        ], np.int32)


    def bbox_zone_overlap_ratio(self, bbox, polygon, frame_shape):
        h_img, w_img = frame_shape[:2]

        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        zone_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        x, y, w, h = bbox
        cv2.rectangle(
            bbox_mask,
            (x, y),
            (x + w, y + h),
            255,
            -1
        )

        cv2.fillPoly(zone_mask, [polygon], 255)

        intersection = cv2.bitwise_and(bbox_mask, zone_mask)

        inter_area = cv2.countNonZero(intersection)
        bbox_area = w * h

        if bbox_area == 0:
            return 0.0

        return inter_area / bbox_area
    
    def vertical_clipping_ratio(self, bbox, frame_height):
        x, y, w, h = bbox
        y_top = y
        y_bottom = y + h

        clipped_top = max(0, -y_top)
        clipped_bottom = max(0, y_bottom - frame_height)

        clipped_pixels = clipped_top + clipped_bottom

        return clipped_pixels / h if h > 0 else 0.0

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def get_zone(self, bbox, width, height):
        x, y, w, h = bbox
        bottom_center = (int(x + w/2), int(y + h))

        if self.point_in_polygon(bottom_center, self.detection_zone):
            if bottom_center[0] < width // 2:
                return "LEFT"
            else:
                return "RIGHT"

        return "OUTSIDE"

    # --------------------------------------------------------------------
    # PROXIMITY + FPS
    # --------------------------------------------------------------------
    def calculate_proximity(self, bbox, frame_shape, track_id=None):
        h_img, w_img = frame_shape[:2]

        overlap = self.bbox_zone_overlap_ratio(
            bbox,
            self.detection_zone,
            frame_shape
        )

        temporal = 0.0
        if track_id is not None:
            temporal = self.temporal_depth_score(track_id, bbox, h_img)

        # Fused proximity score
        score = (
            overlap * 0.6 +
            temporal * 0.4
        )

        if score > self.danger_threshold:
            level, color = "DANGER", (0, 0, 255)
        elif score > self.warning_threshold:
            level, color = "WARNING", (0, 165, 255)
        elif score > self.detected_threshold:
            level, color = "DETECTED", (0, 255, 255)
        else:
            level, color = "SAFE", (0, 255, 0)

        debug_data = {
            "overlap": overlap,
            "temporal": temporal,
            "score": score
        }

        return level, color, score, debug_data


    def draw_proximity_debug(self, frame, bbox, debug_data):
        x, y, w, h = bbox

        # Small semi-transparent panel
        panel_w, panel_h = 180, 70
        px = x
        py = max(0, y - panel_h - 5)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        lines = [
            f"Overlap : {debug_data['overlap']:.3f}",
            f"Temp    : {debug_data['temporal']:.3f}",
            f"Score   : {debug_data['score']:.3f}",
        ]


        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (px + 6, py + 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    def temporal_depth_score(self, track_id, bbox, frame_height):
        x, y, w, h = bbox
        area = w * h
        bottom = y + h
        now = time.time()

        if track_id not in self.temporal_memory:
            self.temporal_memory[track_id] = {
                "h": h,
                "area": area,
                "bottom": bottom,
                "time": now,
                "ema": 0.0
            }
            return 0.0

        prev = self.temporal_memory[track_id]
        dt = now - prev["time"]
        if dt <= 0:
            return prev["ema"]

        dh = max(0, (h - prev["h"]) / frame_height)
        darea = max(0, (area - prev["area"]) / (frame_height ** 2))
        dbottom = max(0, (bottom - prev["bottom"]) / frame_height)

        raw_score = (dh * 0.5) + (darea * 0.3) + (dbottom * 0.2)

        # EMA smoothing
        ema = self.temporal_alpha * raw_score + (1 - self.temporal_alpha) * prev["ema"]

        self.temporal_memory[track_id] = {
            "h": h,
            "area": area,
            "bottom": bottom,
            "time": now,
            "ema": ema
        }

        return min(ema, 1.0)



    def calculate_fps(self):
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now
        self.fps_history.append(fps)
        return sum(self.fps_history) / len(self.fps_history)

    # --------------------------------------------------------------------
    # DRAWING UTILITIES
    # --------------------------------------------------------------------
    def draw_text_bg(self, frame, text, pos, color, scale=0.7, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        cv2.rectangle(frame, (pos[0]-5, pos[1]-size[1]-5),
                      (pos[0]+size[0]+5, pos[1]+5), (0,0,0), -1)
        cv2.putText(frame, text, pos, font, scale, color, thickness)

    def draw_zone_overlay(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.detection_zone], (0, 140, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [self.detection_zone], True, (0, 140, 255), 3)

    # --------------------------------------------------------------------
    # MAIN FRAME PROCESS
    # --------------------------------------------------------------------
    def process_frame(self, frame):
        h, w = frame.shape[:2]

        if self.detection_zone is None:
            self.create_zone_polygons(w, h)

        # results = self.model.track(frame, conf=0.5, iou=0.5, persist=True, verbose=False)
        results = self.model.track(
            frame,
            conf=0.5,
            iou=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )



        warnings = {"LEFT": [], "RIGHT": []}

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else None
                # track_id = None

                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                label = self.model.names[cls]

                zone = self.get_zone(bbox, w, h)
                # proximity, color, score = self.calculate_proximity(bbox, h)
                proximity, color, score, debug_data = self.calculate_proximity(bbox, frame.shape,track_id)

                # Draw bounding boxes
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)

                txt = f"{label} {proximity} ({conf:.2f})"
                if track_id: txt += f" ID:{track_id}"
                self.draw_text_bg(frame, txt, (bbox[0], bbox[1]-10), color)
                if self.debug:
                    self.draw_proximity_debug(frame, bbox, debug_data)


                if zone in warnings:
                    warnings[zone].append({"proximity": proximity, "score": score})

        # Draw detection zone
        self.draw_zone_overlay(frame)

        # Draw FPS
        fps = self.calculate_fps()
        self.draw_text_bg(frame, f"FPS: {int(fps)}", (10, 40), (255,255,255))

        return frame

    # --------------------------------------------------------------------
    # VIDEO / STREAM READER
    # --------------------------------------------------------------------
    def run_stream(self, source=0, save_output=False, output_path="output.mp4"):
        print("Starting stream...")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("ERROR: Cannot open video source")
            return

        # Create resizable window
        cv2.namedWindow("Blind Spot Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Blind Spot Detection", 960, 540)
        # Video writer (optional)
        writer = None
        if save_output:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video / stream")
                break

            processed = self.process_frame(frame)

            cv2.imshow("Blind Spot Detection", processed)

            if save_output:
                writer.write(processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BlindSpotDetector(
        model_path="best_yolov11n_BDD100K_50.pt",
        use_gpu=True
    )

    video_path = "Left_Camera.MOV"  # ganti dengan path video kamu
    detector.run_stream(
        source=video_path,
        save_output=True,
        output_path="blindspot_result_left.mp4"
    )
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class BlindSpotDetector:
    def __init__(self, model_path='best_yolov11n_BDD100K_50.pt', use_gpu=True):
        self.debug = True

        self.model = YOLO(model_path)

        if use_gpu:
            try:
                self.model.to('cuda')
                print("✓ Using GPU")
            except:
                print("⚠ GPU unavailable, using CPU")

        # Thresholds
        self.danger_threshold = 0.4
        self.warning_threshold = 0.25

        self.prev_gray = None
        self.flow_memory = {}   # track_id → smoothed (dx, dy)

        self.fps_history = deque(maxlen=20)
        self.prev_time = time.time()

        self.detection_zone = None

    # ------------------------------------------------------------
    # ZONE
    # ------------------------------------------------------------
    def create_zone_polygons(self, w, h):
        self.detection_zone = np.array([
            [int(w*0.15), int(h*0.8)],
            [int(w*0.15), int(h*0.5)],
            [int(w*0.8), int(h*0.5)],
            [int(w*0.8), int(h*0.8)]
        ], np.int32)

    def is_horizontally_in_zone(self, bbox, polygon):
        """
        Checks if bbox center-x lies inside zone horizontally
        """
        x, y, w, h = bbox
        bbox_cx = x + w * 0.5

        zone_x = polygon[:, 0]
        zone_left = np.min(zone_x)
        zone_right = np.max(zone_x)

        return zone_left <= bbox_cx <= zone_right


    def bbox_zone_vertical_overlap(self, bbox, polygon, frame_shape):
        """
        Returns [0–1] based only on vertical (Y-axis) overlap
        """
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


    # ------------------------------------------------------------
    # PROXIMITY SCORE
    # ------------------------------------------------------------

    def calculate_proximity(self, bbox, frame_shape):
        # Horizontal gating (FIX)
        if not self.is_horizontally_in_zone(bbox, self.detection_zone):
            return "SAFE", (0, 255, 0), 0.0, {
                "vertical_overlap": 0.0,
                "score": 0.0,
                "gated": True
            }

        vertical_overlap = self.bbox_zone_vertical_overlap(
            bbox, self.detection_zone, frame_shape
        )

        # Emphasize near-bottom danger
        score = vertical_overlap ** 1.5

        if score > self.danger_threshold:
            level, color = "DANGER", (0, 0, 255)
        elif score > self.warning_threshold:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)

        debug = {
            "vertical_overlap": vertical_overlap,
            "score": score,
            "gated": False
        }

        return level, color, score, debug



    # ------------------------------------------------------------
    # DRAW DEBUG
    # ------------------------------------------------------------
    def draw_debug(self, frame, bbox, d):
        x, y, _, _ = bbox
        lines = [
            f"V_overlap: {d['vertical_overlap']:.2f}",
            f"Score: {d['score']:.2f}"
        ]

        for i, t in enumerate(lines):
            cv2.putText(
                frame, t,
                (x, y - 10 - i*16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255,255,255), 1
            )

    # ------------------------------------------------------------
    def draw_zone(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.detection_zone], (0,140,255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [self.detection_zone], True, (0,140,255), 2)

    def calculate_fps(self):
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now
        self.fps_history.append(fps)
        return sum(self.fps_history) / len(self.fps_history)

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.detection_zone is None:
            self.create_zone_polygons(w, h)

        results = self.model.track(
            frame,
            conf=0.5,
            iou=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2-x1, y2-y1)

                level, color, score, dbg = self.calculate_proximity(
                    bbox, frame.shape
                )

                cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
                cv2.putText(
                    frame, f"{level}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2
                )

                if self.debug:
                    self.draw_debug(frame, bbox, dbg)

        self.draw_zone(frame)

        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS:{int(fps)}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        self.prev_gray = gray
        return frame

    # ------------------------------------------------------------
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

    video_path = "Recording 2025-11-17 161259.mp4"
    detector.run_stream(
        source=video_path,
        save_output=True,
        output_path="blindspot_result_front.mp4"
    )
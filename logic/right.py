import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class BlindSpotDetector:
    def __init__(self, model_path='best_yolov11n_BDD100K_50.pt', use_gpu=True):
        self.debug = False

        # Camera pointing ~30° forward from right side
        self.camera_angle_deg = 30
        self.camera_angle_rad = np.deg2rad(self.camera_angle_deg)


        self.model = YOLO(model_path)

        if use_gpu:
            try:
                self.model.to('cuda')
                print("✓ Using GPU")
            except:
                print("⚠ GPU unavailable, using CPU")

        # Thresholds
        self.danger_threshold = 0.5
        self.warning_threshold = 0.30

        self.prev_gray = None
        self.flow_memory = {}   # track_id → smoothed (dx, dy)

        self.fps_history = deque(maxlen=20)
        self.prev_time = time.time()

        self.detection_zone = None

    # ------------------------------------------------------------
    # ZONE
    # ------------------------------------------------------------
    def create_zone_polygons(self, width, height):
        self.detection_zone = np.array([
            [int(width*0.75), int(height*1.3)],
            [int(width*0.5), int(height*1.1)],
            [int(width*0.80), int(height*0.65)],
            [int(width*1.00), int(height*0.7)]
        ], np.int32)

    def bbox_zone_overlap_ratio(self, bbox, polygon, frame_shape):
        h_img, w_img = frame_shape[:2]
        mask_bbox = np.zeros((h_img, w_img), np.uint8)
        mask_zone = np.zeros((h_img, w_img), np.uint8)

        x, y, w, h = bbox
        cv2.rectangle(mask_bbox, (x, y), (x+w, y+h), 255, -1)
        cv2.fillPoly(mask_zone, [polygon], 255)

        inter = cv2.bitwise_and(mask_bbox, mask_zone)
        inter_area = cv2.countNonZero(inter)
        zone_area = cv2.countNonZero(mask_zone)
        bbox_area = w * h

        return inter_area / zone_area


    # ------------------------------------------------------------
    # OPTICAL FLOW
    # ------------------------------------------------------------
    def compute_lateral_flow(self, prev_gray, curr_gray, bbox, track_id):
        x, y, w, h = bbox
        h_img, w_img = curr_gray.shape[:2]

        x = max(0, x); y = max(0, y)
        w = min(w, w_img-x); h = min(h, h_img-y)
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
        if track_id not in self.flow_memory:
            self.flow_memory[track_id] = (dx, dy)
        else:
            px, py = self.flow_memory[track_id]
            dx = 0.7 * px + 0.3 * dx
            dy = 0.7 * py + 0.3 * dy
            self.flow_memory[track_id] = (dx, dy)

        return dx, dy

    # ------------------------------------------------------------
    # PROXIMITY SCORE
    # ------------------------------------------------------------

    def calculate_proximity(self, bbox, frame_shape, dx, dy):
        overlap = self.bbox_zone_overlap_ratio(
            bbox, self.detection_zone, frame_shape
        )

        # Blind-spot direction (tune if needed)
        blindspot_dir = np.array([0.4, 1])
        blindspot_dir /= np.linalg.norm(blindspot_dir)

        motion_vec = np.array([dx, dy])
        proj = np.dot(motion_vec, blindspot_dir)

        # Only care about motion INTO blind spot
        motion_score = max(0.0, proj)
        motion_score = np.clip(motion_score / 5.0, 0.0, 1.0)


        score = 0.6 * overlap + 0.4 * motion_score

        if score > self.danger_threshold and overlap > 0.6:
            level, color = "DANGER", (0, 0, 255)
        elif score > self.warning_threshold and overlap > 0.3:
            level, color = "WARNING", (0, 165, 255)
        else:
            level, color = "SAFE", (0, 255, 0)

        # cv2.arrowedLine(
        #     frame,
        #     (50, 50),
        #     (int(50 + blindspot_dir[0]*50), int(50 + blindspot_dir[1]*50)),
        #     (0,255,0), 2
        # )

        debug = {
            "overlap": overlap,
            "dx": dx,
            "dy": dy,
            "motion": motion_score,
            "score": score
        }

        return level, color, score, debug


    # ------------------------------------------------------------
    # DRAW DEBUG
    # ------------------------------------------------------------
    def draw_debug(self, frame, bbox, d):
        x, y, _, _ = bbox
        lines = [
            f"Overlap: {d['overlap']:.2f}",
            f"dx: {d['dx']:.2f}",
            f"dy: {d['dy']:.2f}",
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

                dx, dy = self.compute_lateral_flow(
                    self.prev_gray, gray, bbox, track_id
                )

                level, color, score, dbg = self.calculate_proximity(
                    bbox, frame.shape, dx, dy
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

    video_path = "Right_Camera.MOV"
    detector.run_stream(
        source=video_path,
        save_output=True,
        output_path="blindspot_result_right.mp4"
    )
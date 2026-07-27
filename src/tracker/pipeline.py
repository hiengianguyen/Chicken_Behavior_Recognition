import cv2

from pathlib import Path
from tqdm import tqdm
import time

from .detector import ChickenDetector

from src.trajectory.manager import TrajectoryManager
from src.trajectory.models import TrajectoryRecord
from src.trajectory.exporter import TrajectoryExporter

class TrackingPipeline:

    def __init__(self,
                 model_path,
                 video_path):

        self.detector = ChickenDetector(model_path)

        self.video_path = video_path

        self.manager = TrajectoryManager()

    def run(self, output_csv: str):

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_index = 0

        start_time = time.time()

        print(f"Video: {self.video_path}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")

        with tqdm(total=total_frames, desc="Tracking") as pbar:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = frame_index / fps

                results = self.detector.detect(frame)

                self._process_results(
                    results,
                    frame_index,
                    timestamp
                )

                frame_index += 1

                pbar.update(1)

        cap.release()

        output_path = Path(output_csv)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        exporter = TrajectoryExporter(
            self.manager
        )

        exporter.export_csv(output_csv)

        elapsed = time.time() - start_time

        print("=" * 40)
        print("Tracking Finished")
        print(f"Frames processed : {frame_index}")
        print(f"Time             : {elapsed:.2f} s")
        print(f"Average FPS      : {frame_index / elapsed:.2f}")
        print(f"Saved to         : {output_csv}")
        print("=" * 40)

    def _process_results(
            self,
            results,
            frame,
            timestamp):

        result = results[0]

        if result.boxes.id is None:
            return

        boxes = result.boxes

        for box in boxes:

            track_id = int(box.id.item())

            x1, y1, x2, y2 = box.xyxy[0]

            x = float((x1 + x2) / 2)

            y = float((y1 + y2) / 2)

            width = float(x2 - x1)

            height = float(y2 - y1)

            confidence = float(box.conf.item())

            class_id = int(box.cls.item())

            record = TrajectoryRecord(

                frame=frame,

                timestamp=timestamp,

                track_id=track_id,

                x=x,

                y=y,

                width=width,

                height=height,

                confidence=confidence,

                class_id=class_id

            )

            self.manager.add_record(record)
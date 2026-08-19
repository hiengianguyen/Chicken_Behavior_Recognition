from ultralytics import YOLO

from .models import Track


class ChickenTracker:

    def __init__(
        self,
        model_path="models/best.pt",
        tracker_cfg="bytetrack.yaml",
        conf=0.4,
    ):

        self.model = YOLO(model_path)

        self.tracker_cfg = tracker_cfg

        self.conf = conf

    def update(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            verbose=False,
        )

        result = results[0]

        detection_count = 0
        track_count = 0

        if result.boxes is not None:
            detection_count = len(result.boxes)

            if result.boxes.id is not None:
                track_count = len(result.boxes.id)

        print(
            f"Detection: {detection_count} | "
            f"Tracks: {track_count} | "
            f"IDs: {[int(x) for x in result.boxes.id.tolist()] if result.boxes.id is not None else []}"
        )

        tracks = []

        if result.boxes is None:
            return tracks

        if result.boxes.id is None:
            return tracks

        for box, track_id in zip(result.boxes, result.boxes.id):

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    x=(x1 + x2) / 2,
                    y=(y1 + y2) / 2,
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                )
            )

        return tracks

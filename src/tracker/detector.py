from ultralytics import YOLO


class ChickenDetector:

    def __init__(self, model_path: str):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
            conf=0.25,
        )

        return results

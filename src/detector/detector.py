from ultralytics import YOLO


class ChickenDetector:

    def __init__(self, model_path="models/yolo_best.pt", conf=0.4, classes=None):

        self.model = YOLO(model_path)

        self.conf = conf

        self.classes = classes

    def detect(self, frame):

        results = self.model.predict(
            source=frame, conf=self.conf, classes=self.classes, verbose=False
        )

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "class_id": class_id,
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    }
                )

        return detections


from ultralytics import YOLO


class ChickenDetector:

    def __init__(self, model_path="models/yolo_best.pt", conf=0.4, classes=None):

        self.model = YOLO(model_path)

        self.conf = conf

        self.classes = classes

    def detect(self, frame):

        results = self.model.predict(
            source=frame, conf=self.conf, classes=self.classes, verbose=False
        )

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "class_id": class_id,
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    }
                )

        return detections

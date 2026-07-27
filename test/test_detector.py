import cv2

from src.detector.detector import ChickenDetector

detector = ChickenDetector(model_path="models/best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detector.detect(frame)

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f'{det["confidence"]:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()

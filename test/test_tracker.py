import cv2

from src.main_tracker.tracker import ChickenTracker

tracker = ChickenTracker()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera device")

# Keep camera resolution stable to avoid stretched / squeezed frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracker", width, height)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    tracks = tracker.update(frame)

    for track in tracks:

        x1, y1, x2, y2 = track.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Keep the display size aligned to the actual camera frame to avoid distortion.
    cv2.imshow("Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

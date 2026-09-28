import cv2


class Visualizer:

    COLORS = {
        "Normal": (0, 255, 0),
        "Standing": (0, 0, 255),
        "Unknown": (255, 255, 255),
    }

    def draw(self, frame, track, record):

        x1, y1, x2, y2 = track.bbox

        color = self.COLORS.get(record.prediction, self.COLORS["Unknown"])

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (int(x1), int(y1) - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"{record.confidence*100:.1f}%",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def draw_many(self, frame, tracks, records):

        record_map = {r.track_id: r for r in records}

        for track in tracks:

            track_id = track.track_id

            if track_id not in record_map:
                continue

            self.draw(frame, track, record_map[track_id])

    def draw_statistics(self, frame, records, fps):
        cv2.putText(
            frame,
            f"FPS : {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

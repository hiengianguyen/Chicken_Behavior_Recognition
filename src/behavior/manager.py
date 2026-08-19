from .models import BehaviorRecord

from .features.speed import SpeedFeature
from .features.acceleration import AccelerationFeature
from .features.standing import StandingFeature


class BehaviorManager:

    def __init__(self):

        self.speed = SpeedFeature()

        self.acceleration = AccelerationFeature()

        self.standing = StandingFeature()

    def update(self, frame, timestamp, tracks):
        """
        tracks:

        [
            {
                "track_id":1,
                "x":...,
                "y":...
            },
            ...
        ]
        """

        records = []

        frame_center_x = sum(t.x for t in tracks) / len(tracks) if tracks else 0.0
        frame_center_y = sum(t.y for t in tracks) / len(tracks) if tracks else 0.0

        for track in tracks:

            track_id = track.track_id

            x = track.x

            y = track.y

            speed = self.speed.compute(track_id, x, y, timestamp)

            acceleration = self.acceleration.compute(track_id, speed, timestamp)

            standing_time = self.standing.compute(track_id, speed, timestamp)

            direction = 0.0
            if hasattr(track, "dx"):
                direction = track.dx
            elif hasattr(track, "direction"):
                direction = track.direction

            record = BehaviorRecord(
                frame=frame,
                timestamp=timestamp,
                track_id=track_id,
                x=x,
                y=y,
                speed=speed,
                acceleration=acceleration,
                direction=direction,
                standing_time=standing_time,
                center_x=frame_center_x,
                center_y=frame_center_y,
            )

            records.append(record)

        return records

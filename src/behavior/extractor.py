import pandas as pd
import numpy as np

from .models import BehaviorRecord
from .smoother import TrajectorySmoother
from .features.speed import SpeedFeature
from .features.acceleration import AccelerationFeature
from .features.standing import StandingFeature
# from .features.distance_center import DistanceCenterFeature


class BehaviorExtractor:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.smoother = TrajectorySmoother(
            alpha=0.3
        )

        grouped = self.df.groupby("track_id")

        smooth_tracks = []

        for _, track_df in grouped:

            smooth_tracks.append(

                self.smoother.smooth_track(track_df)

            )

        self.df = pd.concat(
            smooth_tracks,
            ignore_index=True
        )

        self.speed = SpeedFeature()

        self.acceleration = AccelerationFeature()

        self.frame_statistics = {}

        self._build_frame_statistics()

        # self.distance_center = DistanceCenterFeature()

        self.standing = StandingFeature(
            speed_threshold=5
        )

        # self.direction = DirectionFeature()

    def extract(self):

        results = []

        grouped = self.df.groupby("track_id")

        for track_id, track_df in grouped:

            track_df = track_df.sort_values("frame")

            if len(track_df) < 2:
                continue

            previous = track_df.iloc[0]

            previous_record = None

            for i in range(1, len(track_df)):

                current = track_df.iloc[i]

                record = BehaviorRecord(

                    frame=int(current["frame"]),

                    timestamp=float(current["timestamp"]),

                    track_id=int(track_id),

                    x=float(current["x"]),

                    y=float(current["y"])

                )

                # -------- Feature 1 --------
                record.speed = self.speed.calculate(
                    previous,
                    current
                )

                # -------- Feature 2 --------
                record.acceleration = self.acceleration.calculate(
                    previous_record,
                    record
                )

                # -------- Feature 3 --------
                record.standing_time = self.standing.calculate(
                    previous_record,
                    record
                )
                
                # stats = self.frame_statistics[
                #     record.frame
                # ]

                # distance, normalized = self.distance_center.calculate(

                #     current,

                #     stats["center_x"],

                #     stats["center_y"],

                #     stats["mean_distance"]

                # )

                # record.distance_to_center = distance

                # record.normalized_distance = normalized


                # record.direction = self.direction.calculate(
                #     previous,
                #     current
                # )

                results.append(record)

                previous = current

                previous_record = record

        return results
    
    def _build_frame_statistics(self):

        grouped = self.df.groupby("frame")

        for frame, frame_df in grouped:

            center_x = frame_df["x"].mean()

            center_y = frame_df["y"].mean()

            distances = np.sqrt(
                (frame_df["x"] - center_x) ** 2 +
                (frame_df["y"] - center_y) ** 2
            )

            self.frame_statistics[int(frame)] = {

                "center_x": float(center_x),

                "center_y": float(center_y),

                "mean_distance": float(distances.mean())

            }
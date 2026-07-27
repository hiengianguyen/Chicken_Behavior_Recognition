from typing import Dict
from .models import Trajectory, TrajectoryRecord

class TrajectoryManager:
    """
    Quản lý toàn bộ trajectory của các đối tượng đang được theo dõi.
    """

    def __init__(self):
        self.trajectories: Dict[int, Trajectory] = {}

    def add_record(self, record: TrajectoryRecord):
        """
        Thêm một record mới vào trajectory tương ứng.
        """

        track_id = record.track_id

        if track_id not in self.trajectories:
            self.trajectories[track_id] = Trajectory(track_id)

        self.trajectories[track_id].add_record(record)

    def get_trajectory(self, track_id: int):
        return self.trajectories.get(track_id)

    def get_all(self):
        return self.trajectories

    def clear(self):
        self.trajectories.clear()
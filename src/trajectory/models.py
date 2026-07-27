from dataclasses import dataclass, asdict, field
from typing import List
import math

@dataclass
class TrajectoryRecord:
    """
    Lưu thông tin của một đối tượng (gà) trong một frame.
    """

    frame: int
    timestamp: float

    track_id: int

    x: float
    y: float

    width: float
    height: float

    confidence: float

    class_id: int

    def to_dict(self):
        return asdict(self)
    
    def first_record(self):
        if not self.records:
            return None

        return self.records[0]
    
    def duration(self):
        if len(self.records) < 2:
            return 0.0

        return (
            self.records[-1].timestamp
            - self.records[0].timestamp
        )
    
    def total_distance(self):
        if len(self.records) < 2:
            return 0.0

        distance = 0.0

        for i in range(1, len(self.records)):

            prev = self.records[i - 1]
            curr = self.records[i]

            distance += math.dist(
                (prev.x, prev.y),
                (curr.x, curr.y)
            )

        return distance
    
    def mean_speed(self):
        duration = self.duration()

        if duration == 0:
            return 0.0

        return self.total_distance() / duration
    
    def displacement(self):
        if len(self.records) < 2:
            return 0.0

        first = self.records[0]
        last = self.records[-1]

        return math.dist(
            (first.x, first.y),
            (last.x, last.y)
        )
    
    def movement_ratio(self):
        total = self.total_distance()

        if total == 0:
            return 0.0

        return self.displacement() / total
    
    def bounding_box_area_mean(self):
        if not self.records:
            return 0.0

        areas = [
            r.width * r.height
            for r in self.records
        ]

        return sum(areas) / len(areas)

@dataclass
class Trajectory:
    """
    Lưu toàn bộ lịch sử di chuyển của một track_id.
    """

    track_id: int

    records: List[TrajectoryRecord] = field(default_factory=list)

    def add_record(self, record: TrajectoryRecord):
        self.records.append(record)

    def last_record(self):
        if not self.records:
            return None

        return self.records[-1]

    def __len__(self):
        return len(self.records)
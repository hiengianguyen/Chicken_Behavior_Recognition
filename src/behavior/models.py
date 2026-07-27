from dataclasses import dataclass, asdict


@dataclass
class BehaviorRecord:
    frame: int
    timestamp: float
    track_id: int
    x: float
    y: float
    speed: float = 0.0
    acceleration: float = 0.0
    direction: float = 0.0
    standing_time: float = 0.0
    prediction: str = ""
    confidence: float = 0.0

    def to_dict(self):
        return asdict(self)

    def to_feature(self):
        return [
            self.timestamp,
            self.x,
            self.y,
            self.speed,
            self.acceleration,
            self.direction,
            self.standing_time,
        ]

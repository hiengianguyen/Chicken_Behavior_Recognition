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
    separation_time: float = 0.0
    # distance_to_center: float = 0.0
    # normalized_distance: float = 0.0
    standing_time: float = 0.0
    
    def to_dict(self):
        return asdict(self)
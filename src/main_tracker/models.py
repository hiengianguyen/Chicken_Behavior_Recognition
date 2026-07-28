from dataclasses import dataclass


@dataclass
class Track:

    track_id: int

    bbox: tuple[int, int, int, int]

    x: float

    y: float

    confidence: float

    class_id: int

import math


class SpeedFeature:

    def __init__(self):
        self.last_positions = {}

    def calculate(self, previous, current):

        dx = current["smooth_x"] - previous["smooth_x"]

        dy = current["smooth_y"] - previous["smooth_y"]

        dt = current["timestamp"] - previous["timestamp"]

        if dt <= 0:
            return 0.0

        distance = math.sqrt(dx * dx + dy * dy)

        return distance / dt

    def compute(self, track_id, x, y, timestamp):
        previous = self.last_positions.get(track_id)

        self.last_positions[track_id] = {
            "x": x,
            "y": y,
            "timestamp": timestamp,
        }

        if previous is None:
            return 0.0

        dx = x - previous["x"]
        dy = y - previous["y"]
        dt = timestamp - previous["timestamp"]

        if dt <= 0:
            return 0.0

        distance = math.sqrt(dx * dx + dy * dy)
        return float(distance / dt)

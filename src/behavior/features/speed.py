import math


class SpeedFeature:

    def calculate(self, previous, current):

        dx = current["smooth_x"] - previous["smooth_x"]

        dy = current["smooth_y"] - previous["smooth_y"]

        dt = current["timestamp"] - previous["timestamp"]

        if dt <= 0:
            return 0.0

        distance = math.sqrt(dx * dx + dy * dy)

        return distance / dt

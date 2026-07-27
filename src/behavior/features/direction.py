import math


class DirectionFeature:

    def calculate(
        self,
        previous,
        current
    ):

        dx = current["x"] - previous["x"]

        dy = current["y"] - previous["y"]

        return math.degrees(
            math.atan2(
                dy,
                dx
            )
        )
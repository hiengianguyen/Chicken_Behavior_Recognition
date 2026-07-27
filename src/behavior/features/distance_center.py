import math

class DistanceCenterFeature:

    def calculate(

        self,

        current,

        center_x,

        center_y,

        mean_distance

    ):

        dx = current["smooth_x"] - center_x

        dy = current["smooth_y"] - center_y

        distance = math.sqrt(

            dx * dx +

            dy * dy

        )

        if mean_distance == 0:

            normalized = 0.0

        else:

            normalized = distance / mean_distance

        return distance, normalized
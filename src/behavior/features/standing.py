class StandingFeature:

    def __init__(

        self,

        speed_threshold=5.0

    ):

        self.speed_threshold = speed_threshold

    def calculate(

        self,

        previous_record,

        current_record

    ):

        if current_record.speed <= self.speed_threshold:

            if previous_record is None:

                return 0.0

            dt = (

                current_record.timestamp -

                previous_record.timestamp

            )

            return previous_record.standing_time + dt

        return 0.0
class StandingFeature:

    def __init__(self, speed_threshold=5.0):

        self.speed_threshold = speed_threshold
        self.track_state = {}

    def calculate(self, previous_record, current_record):

        if current_record.speed <= self.speed_threshold:

            if previous_record is None:

                return 0.0

            dt = current_record.timestamp - previous_record.timestamp

            return previous_record.standing_time + dt

        return 0.0

    def compute(self, track_id, speed, timestamp):
        state = self.track_state.setdefault(
            track_id,
            {"standing_time": 0.0, "last_timestamp": timestamp},
        )

        if speed <= self.speed_threshold:
            dt = timestamp - state["last_timestamp"]
            if dt < 0:
                dt = 0.0
            state["standing_time"] += dt
            state["last_timestamp"] = timestamp
            return float(state["standing_time"])

        state["standing_time"] = 0.0
        state["last_timestamp"] = timestamp
        return 0.0

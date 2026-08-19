class AccelerationFeature:

    def __init__(self):
        self.last_speeds = {}

    def calculate(self, previous_record, current_record):

        if previous_record is None:
            return 0.0

        dt = current_record.timestamp - previous_record.timestamp

        if dt <= 0:
            return 0.0

        acceleration = (current_record.speed - previous_record.speed) / dt

        return float(acceleration)

    def compute(self, track_id, speed, timestamp):
        previous = self.last_speeds.get(track_id)

        self.last_speeds[track_id] = {
            "speed": speed,
            "timestamp": timestamp,
        }

        if previous is None:
            return 0.0

        dt = timestamp - previous["timestamp"]

        if dt <= 0:
            return 0.0

        return float((speed - previous["speed"]) / dt)

class AccelerationFeature:

    def calculate(self, previous_record, current_record):

        if previous_record is None:
            return 0.0

        dt = current_record.timestamp - previous_record.timestamp

        if dt <= 0:
            return 0.0

        acceleration = (current_record.speed - previous_record.speed) / dt

        return float(acceleration)

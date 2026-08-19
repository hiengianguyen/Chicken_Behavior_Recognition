from datetime import datetime


class StatusManager:

    def __init__(self, max_missed_frames=15):

        self.states = {}

        self.max_missed_frames = max_missed_frames

    def update(self, tracks, behavior_records=None):

        behavior_records = behavior_records or []

        current_ids = set()

        # --------------------------------
        # Behavior lookup
        # --------------------------------

        behavior_map = {}

        for record in behavior_records:

            track_id = record.get("track_id")

            if track_id is not None:
                behavior_map[int(track_id)] = record

        # --------------------------------
        # Update active tracks
        # --------------------------------

        for track in tracks:

            track_id = int(track.track_id)

            current_ids.add(track_id)

            behavior = behavior_map.get(track_id, {})

            if track_id not in self.states:

                self.states[track_id] = {
                    "track_id": track_id,
                    "status": "active",
                    "behavior": "unknown",
                    "confidence": 0.0,
                    "x": track.x,
                    "y": track.y,
                    "bbox": track.bbox,
                    "speed": 0.0,
                    "acceleration": 0.0,
                    "missed_frames": 0,
                    "first_seen": self._now(),
                    "last_seen": self._now(),
                }

            state = self.states[track_id]

            state["status"] = "active"

            state["confidence"] = float(track.confidence)

            state["x"] = float(track.x)
            state["y"] = float(track.y)

            state["bbox"] = track.bbox

            state["missed_frames"] = 0

            state["last_seen"] = self._now()

            # --------------------------------
            # Behavior
            # --------------------------------

            if behavior:

                # Map prediction -> behavior
                if "prediction" in behavior:
                    state["behavior"] = behavior["prediction"]

                # Use LSTM confidence if available, otherwise track confidence
                if "confidence" in behavior and behavior["confidence"] > 0:
                    state["confidence"] = float(behavior["confidence"])

                if "speed" in behavior:
                    state["speed"] = float(behavior["speed"])

                if "acceleration" in behavior:
                    state["acceleration"] = float(behavior["acceleration"])

        # --------------------------------
        # Missing tracks
        # --------------------------------

        for track_id, state in list(self.states.items()):

            if track_id not in current_ids:

                state["missed_frames"] += 1

                if state["missed_frames"] <= self.max_missed_frames:

                    state["status"] = "temporarily_missing"

                else:

                    state["status"] = "lost"

        return self.get_active_states()

    # --------------------------------
    # Get states
    # --------------------------------

    def get_active_states(self):

        return list(self.states.values())

    # --------------------------------
    # Current snapshot
    # --------------------------------

    def snapshot(self):

        return {
            "timestamp": self._now(),
            "total_tracks": len(self.states),
            "active": sum(
                1 for state in self.states.values() if state["status"] == "active"
            ),
            "temporarily_missing": sum(
                1
                for state in self.states.values()
                if state["status"] == "temporarily_missing"
            ),
            "lost": sum(
                1 for state in self.states.values() if state["status"] == "lost"
            ),
            "chickens": self.get_active_states(),
        }

    # --------------------------------
    # Time
    # --------------------------------

    def _now(self):

        return datetime.now().isoformat()

from src.core.context import AppContext

from src.main_tracker.tracker import ChickenTracker
from src.behavior.manager import BehaviorManager
from src.predictor.predictor import BehaviorPredictor
from src.visualization.visualizer import Visualizer

from src.status.manager import StatusManager


class ChickenPipeline:

    def __init__(
        self,
        detector_model="models/yolo_best.pt",
        behavior_model="models/best_model.pt",
    ):

        # =====================================
        # CORE
        # =====================================

        self.context = AppContext()

        # =====================================
        # TRACKING
        # =====================================

        self.tracker = ChickenTracker(model_path=detector_model)

        # =====================================
        # BEHAVIOR
        # =====================================

        self.behavior_manager = BehaviorManager()

        self.predictor = BehaviorPredictor(model_path=behavior_model)

        # =====================================
        # STATUS
        # =====================================

        self.status_manager = StatusManager(max_missed_frames=15)

        # =====================================
        # VISUALIZATION
        # =====================================

        self.visualizer = Visualizer()

    # =========================================
    # MAIN PIPELINE
    # =========================================

    def process(self, frame):

        # -------------------------------------
        # 1. Update context
        # -------------------------------------

        self.context.update()

        # -------------------------------------
        # 2. YOLO + ByteTrack
        # -------------------------------------

        tracks = self.track(frame)
        self.predictor.sync_tracks(track.track_id for track in tracks)

        # -------------------------------------
        # 3. Extract behavior features
        # -------------------------------------

        records = self.analyze(tracks)

        # -------------------------------------
        # 4. Behavior prediction
        # -------------------------------------

        records = self.predict(records)

        # Convert records to dicts for StatusManager
        records_dict = [r.to_dict() for r in records]

        # -------------------------------------
        # 5. Update chicken status
        # -------------------------------------

        statuses = self.status_manager.update(tracks, records_dict)

        # -------------------------------------
        # 6. Render
        # -------------------------------------

        self.render(frame, tracks, records)

        return frame

    # =========================================
    # TRACKING
    # =========================================

    def track(self, frame):

        return self.tracker.update(frame)

    # =========================================
    # BEHAVIOR ANALYSIS
    # =========================================

    def analyze(self, tracks):

        return self.behavior_manager.update(
            self.context.frame_index, self.context.timestamp, tracks
        )

    # =========================================
    # LSTM PREDICTION
    # =========================================

    def predict(self, records):
        """Update predictions for each behavior record using LSTM"""
        predictions = []
        for record in records:
            result = self.predictor.update(record)
            if result:
                # Merge prediction result into record
                record.prediction = result["label"]
                record.confidence = result["confidence"]
                predictions.append(record)
            else:
                # Still add record even if no prediction yet (buffer not ready)
                predictions.append(record)
        return predictions

    # =========================================
    # VISUALIZATION
    # =========================================

    def render(self, frame, tracks, records):

        self.visualizer.draw_many(frame, tracks, records)

        self.visualizer.draw_statistics(frame, records, self.context.fps)

from src.core.context import AppContext

from src.main_tracker.tracker import ChickenTracker
from src.behavior.manager import BehaviorManager
from src.predictor.predictor import BehaviorPredictor
from src.visualization.visualizer import Visualizer


class ChickenPipeline:

    def __init__(
        self,
        detector_model="models/yolo_best.pt",
        behavior_model="models/behavior_best.pt",
    ):

        self.context = AppContext()

        self.tracker = ChickenTracker(model_path=detector_model)

        self.behavior_manager = BehaviorManager()

        self.predictor = BehaviorPredictor(model_path=behavior_model)

        self.visualizer = Visualizer()

    def process(self, frame):

        self.context.update()

        tracks = self.track(frame)

        records = self.analyze(tracks)

        self.predict(records)

        self.render(frame, tracks, records)

        return frame

    def track(self, frame):

        return self.tracker.update(frame)

    def analyze(self, tracks):

        return self.behavior_manager.update(self.context, tracks)

    def predict(self, records):

        self.predictor.predict(self.context, records)

    def render(self, frame, tracks, records):

        self.visualizer.draw_many(frame, tracks, records)

        self.visualizer.draw_statistics(frame, records, self.context.fps)

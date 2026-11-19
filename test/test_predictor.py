from collections import defaultdict, deque
from types import SimpleNamespace

from src.predictor.predictor import BehaviorPredictor


class _Buffer:
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.features = defaultdict(list)

    def update(self, track_id, feature):
        self.features[track_id].append(feature)

    def clear(self, track_id):
        self.features.pop(track_id, None)

    def ready(self, track_id):
        return len(self.features[track_id]) >= self.window_size

    def get_sequence(self, track_id):
        return self.features[track_id]


def test_motion_clears_standing_prediction():
    predictor = BehaviorPredictor.__new__(BehaviorPredictor)
    predictor.frame_counter = defaultdict(int)
    predictor.buffer = _Buffer()
    predictor.prediction_history = defaultdict(lambda: deque(maxlen=5))
    predictor.confidence_history = defaultdict(lambda: deque(maxlen=5))
    predictor.reanalysis_pending = set()
    predictor.predict_interval = 5
    predictor.predict = lambda sequence: {
        "label": "Standing",
        "class": 1,
        "confidence": 0.9,
    }
    predictor.last_prediction = {
        1: {"label": "Standing", "class": 1, "confidence": 0.9}
    }
    predictor.prediction_history[1].append(1)
    predictor.confidence_history[1].append(0.9)
    record = SimpleNamespace(
        track_id=1,
        speed=6.0,
        to_feature=lambda: [],
    )

    result = predictor.update(record)

    assert result == {"label": "Normal", "class": 0, "confidence": 0.0}
    assert not predictor.prediction_history[1]
    assert not predictor.confidence_history[1]
    assert predictor.last_prediction[1] == result
    assert 1 in predictor.reanalysis_pending
    assert not predictor.buffer.features[1]


def test_reanalysis_waits_for_a_fresh_sequence_after_motion():
    predictor = BehaviorPredictor.__new__(BehaviorPredictor)
    predictor.frame_counter = defaultdict(int)
    predictor.buffer = _Buffer(window_size=2)
    predictor.prediction_history = defaultdict(lambda: deque(maxlen=5))
    predictor.confidence_history = defaultdict(lambda: deque(maxlen=5))
    predictor.reanalysis_pending = set()
    predictor.predict_interval = 1
    predictor.last_prediction = {}
    predictor.predict = lambda sequence: {
        "label": "Standing",
        "class": 1,
        "confidence": 0.9,
    }

    def record(speed):
        return SimpleNamespace(
            track_id=1,
            speed=speed,
            to_feature=lambda: [speed],
        )

    assert predictor.update(record(6.0))["label"] == "Normal"
    assert predictor.update(record(0.0))["label"] == "Normal"
    result = predictor.update(record(0.0))

    assert result["label"] == "Standing"
    assert 1 not in predictor.reanalysis_pending

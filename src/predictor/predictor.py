import numpy as np
import torch

from src.models.lstm import ChickenBehaviorLSTM
from src.predictor.buffer import SequenceBuffer

from collections import defaultdict, deque
from collections import Counter


class BehaviorPredictor:

    LABELS = {
        0: "Normal",
        1: "Standing",
        # 2 : "Separation"
    }

    def __init__(
        self,
        model_path="weights/best_model.pt",
        input_size=9,
        num_classes=2,
        window_size=90,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer = SequenceBuffer(window_size)

        self.model = ChickenBehaviorLSTM(
            input_size=input_size, num_classes=num_classes
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.prediction_history = defaultdict(lambda: deque(maxlen=5))

        self.confidence_history = defaultdict(lambda: deque(maxlen=5))

        self.predict_interval = 5

        self.frame_counter = defaultdict(int)

        self.last_prediction = {}

    def update(self, record):
        """
        record phải có:
            record.track_id
            record.to_feature()
        """

        self.frame_counter[record.track_id] += 1

        self.buffer.update(record.track_id, record.to_feature())

        if not self.buffer.ready(record.track_id):
            return None

        if self.frame_counter[record.track_id] % self.predict_interval != 0:

            return self.last_prediction.get(record.track_id, None)

        sequence = self.buffer.get_sequence(record.track_id)

        result = self.predict(sequence)

        result["class"] = self.smooth_prediction(record.track_id, result["class"])

        result["label"] = self.LABELS[result["class"]]

        result["confidence"] = self.smooth_confidence(
            record.track_id, result["confidence"]
        )

        self.last_prediction[record.track_id] = result

        return result

    def smooth_prediction(self, track_id, prediction):

        history = self.prediction_history[track_id]

        history.append(prediction)

        counter = Counter(history)

        return counter.most_common(1)[0][0]

    def smooth_confidence(self, track_id, confidence):

        history = self.confidence_history[track_id]

        history.append(confidence)

        return sum(history) / len(history)

    @torch.no_grad()
    def predict(self, sequence):

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        x = x.to(self.device)

        output = self.model(x)

        prediction = torch.argmax(output, dim=1).item()

        confidence = torch.softmax(output, dim=1)[0][prediction].item()

        return {
            "label": self.LABELS[prediction],
            "class": prediction,
            "confidence": confidence,
        }

from collections import defaultdict, deque
import numpy as np


class SequenceBuffer:
    """
    Lưu chuỗi feature của từng track_id.
    Khi đủ window_size frame sẽ sẵn sàng đưa vào LSTM.
    """

    def __init__(self, window_size=90):

        self.window_size = window_size

        self.buffers = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, track_id, feature):

        self.buffers[track_id].append(feature)

    def ready(self, track_id):

        return len(self.buffers[track_id]) == self.window_size

    def get_sequence(self, track_id):

        if not self.ready(track_id):
            return None

        return np.array(self.buffers[track_id], dtype=np.float32)

    def clear(self, track_id):

        if track_id in self.buffers:
            del self.buffers[track_id]

    def __len__(self):

        return len(self.buffers)

from dataclasses import dataclass
import time


@dataclass
class AppContext:

    frame_index: int = 0

    fps: float = 0.0

    timestamp: float = 0.0

    start_time: float = time.time()

    last_frame_time: float = time.time()

    total_frames: int = 0

    def update(self):

        now = time.time()

        self.timestamp = now

        dt = now - self.last_frame_time

        if dt > 0:

            self.fps = 1.0 / dt

        self.last_frame_time = now

        self.frame_index += 1

        self.total_frames += 1

import threading
import time
from typing import Union


class ProgressTracker:
    def __init__(self, n_steps: Union[None, int]=None, *, log_detailed=False):
        self.n_steps = n_steps
        self.iteration = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

        self.log_detailed = log_detailed
        self.log = {0: 0}

    def update(self, n: int = 1):
        with self.lock:
            self.iteration += n

        if self.log_detailed:
            minute = int((time.time() - self.start_time) // 60) + 1
            self.log[minute] = self.iteration

        # assert (
        #     self.n_steps is None or self.iteration <= self.n_steps
        # ), f"{self.iteration} <= {self.n_steps}"

    def get_progress(self):
        return [self.iteration, self.n_steps]

    def get_elapsed_time(self):
        return time.time() - self.start_time

    def get_log(self):
        if self.log_detailed:
            return list(sorted(self.log.items(), key=lambda x: x[0]))

        minute = int((time.time() - self.start_time) // 60) + 1
        return [(0, 0), (minute, self.iteration)]

import time
from typing import Optional


class StepCounter:
    def __init__(self, reporting_freq: Optional[int] = 10_000):
        """Helper class for counting the number of steps. Prints out the step number and the steps
        per second at each interval, decided by `reporting_freq`.

        Parameters
        ----------
        reporting_freq: Optional[int]
            How often to report the number of steps and the steps per second. Default: 10,000
        """
        self.steps = 0
        self.starting_time = time.time()
        self.reporting_freq = reporting_freq

    def __call__(self):
        """Call method for the StepCounter. Increases the step count and reports it if we've hit
        the reporting frequency."""
        self.steps += 1
        if self.steps % self.reporting_freq == 0:
            steps_per_sec = self.reporting_freq / (time.time() - self.starting_time)
            print(f"\nStep {self.steps}: {steps_per_sec} steps/sec")
            self.starting_time = time.time()

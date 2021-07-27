import sys
from typing import List, Dict
from typing import Optional
import numpy as np


def centered(in_str: str, desired_len: Optional[int] = 0) -> str:
    return in_str.center(max(len(in_str) + 2, desired_len))


class ProgressBar:
    def __init__(self, keys: List[str], max_bar_length: Optional[int] = 40):
        epoch_section = "Epoch"
        bar_section = centered("Progress Bar", max_bar_length - 2)

        # We add keys on a second time for the validation set
        self.keys = [epoch_section, bar_section] + keys + keys
        title_bar = ""
        divider = ""

        section_lengths = {
            "Epoch Info": len(centered("Epoch")) + max_bar_length + 1,
            "Training": sum([len(centered(key)) + 1 for key in keys]) - 1,
            "Validation": sum([len(centered(key)) + 1 for key in keys]) - 1,
        }

        for key in self.keys:
            divider += "+"
            title_bar += "|"

            title_key = centered(key)
            divider += "-" * len(title_key)
            title_bar += title_key
        title_bar += "|"
        divider += "+"
        mode_divider = ""
        for k, v in section_lengths.items():
            mode_divider += "|" + centered(k, v)
        mode_divider += "|"

        self.divider = divider

        self.header = (
            divider + "\n" + mode_divider + "\n" + divider + "\n" + title_bar + "\n" + divider
        )
        self.epoch = 0
        self._header_printed = False
        self._max_length = max_bar_length - 2
        self._training_str = ""

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch
        print("")

    def print_bar(self, progress: float, data: Dict[str, float], train_step: bool) -> None:
        if not self._header_printed:
            print(self.header)
            self._header_printed = True

        # Determine how long the progress bar should be
        bar_length = int(np.max((np.floor(self._max_length * progress), 1)))

        if not train_step:
            bar_length += 1

        bar_char = "-" if train_step else "="
        pointer_char = ">" if train_step else "="
        space_char = " " if train_step else "-"

        # Build the progress bar
        bar = (
            "["
            + bar_char * (bar_length - 1)
            + pointer_char
            + space_char * (self._max_length - bar_length)
            + "]"
        )

        epoch_bar_section = f"\r|{centered(str(self.epoch), len(centered('Epoch')))}|{bar}|"

        if train_step:
            self._training_str = ""
            validation_str = ""
            for k, v in data.items():
                self._training_str += centered(f"{v:0.3f}", len(centered(k))) + "|"
                validation_str += " " * len(centered(k)) + "|"
        else:
            validation_str = ""
            for k, v in data.items():
                validation_str += centered(f"{v:0.3f}", len(centered(k))) + "|"

        sys.stdout.write(epoch_bar_section + self._training_str + validation_str)
        sys.stdout.flush()

    def close(self):
        print(self.divider)

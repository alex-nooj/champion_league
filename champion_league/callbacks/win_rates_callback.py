import collections

from stable_baselines3.common.callbacks import BaseCallback
from tabulate import tabulate


class WinRatesCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super(WinRatesCallback, self).__init__(verbose)
        self.won_battles = 0
        self.battles_finished = 0
        self.stats = {}

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.battles_finished != self.training_env.envs[0].env.n_finished_battles:
            opponent_name = self.training_env.envs[0].env._opponent.username.rsplit(
                " "
            )[0]
            if opponent_name not in self.stats:
                self.stats[opponent_name] = collections.deque(maxlen=100)
            if self.won_battles != self.training_env.envs[0].env.n_won_battles:
                self.stats[opponent_name].append(1)
            else:
                self.stats[opponent_name].append(0)
            self.won_battles = self.training_env.envs[0].env.n_won_battles
            self.battles_finished = self.training_env.envs[0].env.n_finished_battles
            if self.battles_finished % 100 == 0:
                data = [
                    ["Agent", "Score"],
                ] + [[k, sum(v) / 100] for k, v in self.stats.items()]
                print(tabulate(data, headers="firstrow", tablefmt="fancy_grid"))
        return True

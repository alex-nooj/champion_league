from poke_env.environment.battle import Battle


class Rule:
    def __init__(self, weight: float):
        self.weight = weight

    def compute(self, curr_step: Battle) -> float:
        raise NotImplementedError

    @property
    def max(self) -> float:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

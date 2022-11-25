from pathlib import Path


class PokePath:
    def __init__(self, logdir: str, tag: str):
        self.logdir = logdir
        self.tag = tag

    @property
    def league(self) -> Path:
        return Path(self.logdir, "league")

    @property
    def challengers(self) -> Path:
        return Path(self.logdir, "challengers")

    @property
    def exploiters(self) -> Path:
        return Path(self.logdir, "exploiters")

    @property
    def agent(self) -> Path:
        return Path(self.logdir, "challengers", self.tag)

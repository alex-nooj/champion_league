from pathlib import Path


class PokePath:
    def __init__(self, logdir: str, tag: str):
        self._logdir = logdir
        self._tag = tag

    @property
    def league(self) -> Path:
        return Path(self._logdir, "league")

    @property
    def challengers(self) -> Path:
        return Path(self._logdir, "challengers")

    @property
    def exploiters(self) -> Path:
        return Path(self._logdir, "exploiters")

    @property
    def agent(self) -> Path:
        return Path(self._logdir, "challengers", self._tag)

from pathlib import Path
from typing import Optional


def get_save_dir(
    agent_dir: Path, epoch: int, create_dir: Optional[bool] = True
) -> Path:
    tag = agent_dir.name
    save_dir = agent_dir / f"{tag}_{epoch:05d}"

    if create_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_most_recent_epoch(agent_path: Path) -> int:
    epochs = [
        int(str(e).rsplit("_")[-1])
        for e in agent_path.iterdir()
        if e.is_dir() and str(e) != "sl"
    ]

    try:
        return max(epochs)
    except ValueError:
        return 0


class PokePath:
    def __init__(self, logdir: str, tag: str):
        self._logdir = logdir
        self.tag = tag

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
        return Path(self._logdir, "challengers", self.tag)

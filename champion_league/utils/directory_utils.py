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

    return max(epochs)

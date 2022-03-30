from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import Tuple

import torch
from torch import nn

from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir


class BaseNetwork(nn.Module):
    @abstractmethod
    def forward(
        self, x_internals: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def reset(self, device: int) -> Dict[str, torch.Tensor]:
        return {}

    def resume(self, agent_dir: Path):
        try:
            epoch = get_most_recent_epoch(agent_dir)
            save_dir = get_save_dir(agent_dir=agent_dir, epoch=epoch)
            self.load_state_dict(
                torch.load(
                    save_dir / "network.pt", map_location=lambda storage, loc: storage
                ),
            )
        except ValueError:
            if (agent_dir / "sl").is_dir():
                self.load_state_dict(
                    torch.load(
                        agent_dir / "sl" / "best_model.pt",
                        map_location=lambda storage, loc: storage,
                    ),
                )

from pathlib import Path

import torch
from torch import nn

from champion_league.network.ability_network import AbilityNetwork
from champion_league.network.encoder_lstm import EncoderLSTM
from champion_league.network.gated_encoder import GatedEncoder
from champion_league.utils.directory_utils import get_save_dir

NETWORKS = {
    GatedEncoder.__name__: GatedEncoder,
    AbilityNetwork.__name__: AbilityNetwork,
    EncoderLSTM.__name__: EncoderLSTM,
}


def save_network(agent_dir: Path, epoch: int, network: nn.Module):
    save_file = get_save_dir(agent_dir, epoch) / "network.pt"
    torch.save(network.state_dict(), str(save_file))

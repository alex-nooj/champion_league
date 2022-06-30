from typing import Any
from typing import Dict
from typing import Tuple

from torch import nn

from champion_league.network import NETWORKS
from champion_league.preprocessor import Preprocessor


def build_network_and_preproc(args: Dict[str, Any]) -> Tuple[nn.Module, Preprocessor]:
    preprocessor = Preprocessor(args["device"], **args["preprocessor"])
    if args["network"] in args:
        network_args = args[args["network"]]
    else:
        network_args = {}
    network = (
        NETWORKS[args["network"]](
            nb_actions=args["nb_actions"],
            in_shape=preprocessor.output_shape,
            **network_args,
        )
        .eval()
        .to(args["device"])
    )
    return network, preprocessor

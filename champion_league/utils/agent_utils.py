from typing import Any
from typing import Dict
from typing import Tuple

from torch import nn

from champion_league.network import NETWORKS
from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors import PREPROCESSORS


def build_network_and_preproc(args: Dict[str, Any]) -> Tuple[nn.Module, Preprocessor]:
    preprocessor = PREPROCESSORS[args["preprocessor"]](
        args["device"], **args[args["preprocessor"]]
    )
    network = (
        NETWORKS[args["network"]](
            nb_actions=args["nb_actions"],
            in_shape=preprocessor.output_shape,
            **args[args["network"]],
        )
        .eval()
        .to(args["device"])
    )
    return network, preprocessor

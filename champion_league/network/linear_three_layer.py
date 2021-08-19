from collections import OrderedDict

import torch.nn as nn
from adept.utils.util import DotDict


class LinearThreeLayer(nn.Module):
    def __init__(self, nb_actions, in_shape, device):
        super(LinearThreeLayer, self).__init__()
        self.device = device
        self.linears = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("linear1", nn.Linear(in_shape[0] * in_shape[1], 128, bias=False)),
                    ("norm1", nn.BatchNorm1d(128)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(128, 128, bias=False)),
                    ("norm2", nn.BatchNorm1d(128)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(128, 128, bias=False)),
                    ("norm3", nn.BatchNorm1d(128)),
                    ("relu3", nn.ReLU()),
                ]
            )
        )
        self.outputs = nn.ModuleDict(
            {
                "critic": nn.Linear(128, 1, bias=True),
                "action": nn.Sequential(
                    OrderedDict(
                        [
                            ("linear_out", nn.Linear(128, nb_actions, bias=True)),
                            ("softmax", nn.Softmax(dim=-1)),
                        ]
                    )
                ),
            }
        )

    def forward(self, x):
        x = self.linears(x)

        outputs = {key: self.outputs[key](x) for key in self.outputs}

        return outputs


def build_from_args(args: DotDict) -> LinearThreeLayer:
    return LinearThreeLayer(args.nb_actions, args.in_shape, args.device)

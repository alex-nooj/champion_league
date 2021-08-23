import torch
from champion_league.utils.directory_utils import DotDict
from champion_league.agent.ppo.ppo_agent import PPOAgent


def build_agent_from_args(args: DotDict, network: torch.nn.Module) -> PPOAgent:
    return PPOAgent(
        device=args.device,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=args.logdir,
        tag=args.tag,
        mini_epochs=args.mini_epochs,
    )

from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import torch
from omegaconf import OmegaConf
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.agent.scripted.base_scripted import BaseScripted
from champion_league.network import NETWORKS
from champion_league.preprocessor import Preprocessor
from champion_league.teams.team_builder import AgentTeamBuilder


class OpponentPlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        opponent: Union[RLOpponent, BaseScripted],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opponent = opponent

    @classmethod
    def from_path(
        cls, path: Path, device: Optional[int] = None, **kwargs
    ) -> "OpponentPlayer":
        """Creates the player from a given path

        Parameters
        ----------
        path: str
            The path to the desired agents directory
        device: Optional[int]
            The GPU to load the agent on. If None, defaults to 0
        kwargs: Dict[Any]
            Keyword arguments to be passed to the Player class

        Returns
        -------
        """
        if device is None:
            device = 0

        args = OmegaConf.to_container(OmegaConf.load(path / "args.yaml"))

        if "scripted" in args:
            opponent = SCRIPTED_AGENTS[args["agent"]]
            team = AgentTeamBuilder()
        else:
            preprocessor = Preprocessor(args["device"], **args["preprocessor"])
            network_args = {}
            if args["network"] in args:
                network_args = args[args["network"]]
            network = NETWORKS[args["network"]](
                nb_actions=args["nb_actions"],
                in_shape=preprocessor.output_shape,
                **network_args,
            ).eval()
            network.load_state_dict(
                torch.load(
                    Path(path, "network.pt"),
                    map_location=lambda storage, loc: storage,
                )
            )

            opponent = RLOpponent(
                network=network,
                preprocessor=preprocessor,
                device=device,
                sample_moves=False,
            )

            team = AgentTeamBuilder(agent_path=path.parent)
        return cls(opponent, battle_format=args["battle_format"], team=team, **kwargs)

    def choose_move(self, battle: Battle) -> BattleOrder:
        """Function that allows the agent to select a move.

        Parameters
        ----------
        battle: Battle
            The current game state.

        Returns
        -------
        BattleOrder
            The action the agent would like to take, in a format readable by Showdown!
        """
        return self.opponent.choose_move(battle)

    @property
    def battle_history(self) -> List[bool]:
        """Returns a list containing the win/loss results of the agent.

        Returns
        -------
        List[bool]
            Contains the win/loss history of the agent.
        """
        return [b.won for b in self._battles.values()]

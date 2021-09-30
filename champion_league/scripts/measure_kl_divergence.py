import asyncio
import copy
import json
import os
from typing import List
from typing import Union

import plotly.express as px
import torch
from poke_env.environment.battle import Battle
from poke_env.player.player import Player

from champion_league.agent.scripted.max_damage_player import MaxDamagePlayer
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.parse_args import parse_args


class KLMaxDamage(MaxDamagePlayer):
    def __init__(self):
        super().__init__(None)
        self.steps = []
        self.step_count = 0

    def choose_move(self, battle: Battle):
        """Chooses a move for the agent. Saves every 10th step.

        Parameters
        ----------
        battle: Battle
            The raw battle output from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        if self.step_count % 10 == 0:
            self.steps.append(copy.deepcopy(battle))
        self.step_count += 1
        return super().choose_move(battle)


async def match(player1: Player, player2: Player, nb_battles: int) -> None:
    """Asynchronous function for having two players battle each other.

    Parameters
    ----------
    player1: Player
        The player that hosts the battles.
    player2: Player
        The player that accepts the battles.
    nb_battles: int
        The number of battles to be played.

    Returns
    -------
    None
    """
    await player1.battle_against(player2, n_battles=nb_battles)


def get_agent_distributions(
    agent_name: str, league_dir: str, battles: List[Battle]
) -> Union[torch.Tensor, None]:
    """Function for getting a specified network's distribution over a batch of game states.

    Parameters
    ----------
    agent_name: str
        The name of the agent to be loaded from `league_dir`.
    league_dir: str
        The path to the league.
    battles: List[Battle]
        A list of Battle Objects from PokeEnv to be predicted over.

    Returns
    -------
    Union[torch.Tensor, None]
        Returns `None` if the agent is scripted, otherwise returns a tensor containing all of the
        action distributions for the agent.
    """
    with open(os.path.join(league_dir, agent_name, "args.json"), "r") as fp:
        args = DotDict(json.load(fp))

    if "scripted" in args:
        return None

    args.resume = False
    network = build_network_from_args(args)
    network.load_state_dict(
        torch.load(
            os.path.join(league_dir, agent_name, "network.pt"),
            map_location=lambda storage, loc: storage,
        )
    )

    preprocessor = build_preprocessor_from_args(args)

    proc_battles = [preprocessor.embed_battle(battle) for battle in battles]
    tensor_battles = {
        k: torch.stack([pb[k] for pb in proc_battles]).squeeze()
        for k in proc_battles[0]
    }

    return network.forward(tensor_battles)["action"]


def measure_league_diversity(nb_battles: int, league_dir: str) -> None:
    """Determines and plots the KL divergence between each agent, as well as the number of times
    two agents select the same move. Used to determine when two agents are near-identical for
    pruning.

    Parameters
    ----------
    nb_battles: int
        The number of battles to be played.
    league_dir: str
        The path to the league.

    Returns
    -------
    None
    """
    player1 = KLMaxDamage()
    player2 = KLMaxDamage()
    asyncio.get_event_loop().run_until_complete(match(player1, player2, nb_battles))

    battles = player1.steps + player2.steps

    agent_predictions = {}
    for agent in os.listdir(league_dir):
        predictions = get_agent_distributions(agent, league_dir, battles)
        if predictions is not None:
            agent_predictions[agent] = predictions.cpu()

    kl_divergence = []
    loss = torch.nn.KLDivLoss()
    for agent1 in agent_predictions:
        kl_div = []
        for agent2 in agent_predictions:
            if agent1 == agent2:
                kl_div.append(0.0)
            else:
                kl_div.append(
                    loss(
                        torch.log(agent_predictions[agent1]),
                        agent_predictions[agent2],
                    ).item()
                )
        kl_divergence.append(kl_div)
    fig = px.imshow(kl_divergence)
    fig.show()

    selected_actions = []
    for agent1 in agent_predictions:
        action_matches = []
        for agent2 in agent_predictions:
            action_matches.append(
                int(
                    torch.mean(
                        (
                            torch.argmax(agent_predictions[agent1], dim=-1)
                            == torch.argmax(agent_predictions[agent2], dim=-1)
                        ).float()
                    )
                    >= 0.5
                )
            )
        selected_actions.append(action_matches)
    fig = px.imshow(selected_actions)
    fig.show()


if __name__ == "__main__":
    measure_league_diversity(**parse_args())

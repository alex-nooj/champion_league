import asyncio
import typing

import torch
from poke_env.environment.battle import Battle
from poke_env.player.player import Player

from champion_league.preprocessor.ops import EmbedMoves
from champion_league.teams.agent_team_builder import AgentTeamBuilder
from champion_league.utils.server_configuration import DockerServerConfiguration


class ScriptedPlayer(Player):
    test_battles = []
    turn_count = 0

    def choose_move(self, battle: Battle):
        self.test_battles.append(copy.deepcopy(battle))
        # If the player can attack, it will
        self.turn_count += 1
        return self.create_order(
            battle.available_switches[self.turn_count % len(battle.available_switches)]
        )


class TestEmbedMoves:
    def get_battles(self) -> typing.List[Battle]:
        player1 = ScriptedPlayer(
            server_configuration=DockerServerConfiguration,
            battle_format="gen8ou",
            team=AgentTeamBuilder(),
        )
        player2 = ScriptedPlayer(
            server_configuration=DockerServerConfiguration,
            battle_format="gen8ou",
            team=AgentTeamBuilder(),
        )
        asyncio.get_event_loop().run_until_complete(
            player1.battle_against(player2, n_battles=1)
        )
        return player1.test_battles

    def test_all_below_one(self):
        battles = self.get_battles()
        op = EmbedMoves(ally=True, in_shape=(0,))
        for battle in battles:
            ret_tensor = op.preprocess(battle, torch.tensor([]))
            violations = (ret_tensor > 1.0).nonzero()
            if len(violations) != 0:
                assert False
        assert True

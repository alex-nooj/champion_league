from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status

from champion_league.env.base.obs_idx import ObsIdx
from champion_league.utils.abilities import ABILITIES


class LSTMNetwork(nn.Module):
    INPUT_SIZE = (1, 12 * 55)

    def __init__(
        self, nb_actions: int,
    ):
        super(LSTMNetwork, self).__init__()

        input_linear_size = 512
        lstm_hidden_size = 512
        self.lstm_hidden_size = lstm_hidden_size
        # self.input_head = nn.Linear(self.INPUT_SIZE[1], input_linear_size)
        # self.input_heads = nn.ModuleDict(
        #     {
        #         f"pokemon_{i}": nn.Linear(self.INPUT_SIZE[1], input_linear_size)
        #         for i in range(self.INPUT_SIZE[0])
        #     }
        # )

        self.lstm = nn.LSTMCell(self.INPUT_SIZE[1], lstm_hidden_size)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.output_layers = nn.ModuleDict(
            {
                "critic": nn.Linear(lstm_hidden_size, 1, bias=True),
                "action": nn.Linear(lstm_hidden_size, nb_actions, bias=True),
            }
        )

    def forward(
        self, xs: torch.Tensor, internals: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        hxs = internals["hx"]
        cxs = internals["cx"]

        # Separate all 12 pokemon into separate tensors to pass into the linear networks
        # separated_inputs = torch.split(xs, 1, dim=1)
        # processed_mons = []
        # for input_mon, key in zip(separated_inputs, self.input_heads):
        #     processed_mons.append(F.relu(self.input_heads[key](input_mon.squeeze(1))))
        # lstm_input = F.relu(self.input_head(xs))
        # lstm_input = torch.cat(processed_mons, dim=1)
        next_hx, next_cx = self.lstm(xs, (hxs, cxs))

        outputs = {
            "critic": self.output_layers["critic"](next_hx),
            "action": F.softmax(self.output_layers["action"](next_hx), dim=-1),
        }

        return outputs, {"hx": next_hx, "cx": next_cx}

    def embed_battle(self, battle: Battle) -> torch.Tensor:
        state = torch.zeros(12, 55)

        # For pokemon in battle:
        all_pokemon = [battle.active_pokemon] + battle.available_switches
        for ix, pokemon in enumerate(all_pokemon):
            # Divide by the number of types + 1 (for no type)
            state[ix, ObsIdx.type1] = pokemon.type_1.value / 19.0
            if pokemon.type_2 is not None:
                state[ix, ObsIdx.type2] = pokemon.type_2.value / 19.0

            if pokemon.ability is not None:
                state[ix, ObsIdx.ability_bit0 : ObsIdx.ability_bit0 + 9] = torch.tensor(
                    ABILITIES[
                        pokemon.ability.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("'", "")
                    ]
                )

            # Blissey has the maximum HP at 714
            if pokemon.current_hp is not None:
                state[ix, ObsIdx.current_hp] = pokemon.current_hp / 714.0

            state[ix, ObsIdx.hp_ratio] = pokemon.current_hp_fraction

            state[ix, ObsIdx.base_hp] = pokemon.base_stats["hp"] / 255.0  # Blissey
            state[ix, ObsIdx.base_atk] = pokemon.base_stats["atk"] / 190.0  # Mega Mewtwo X
            state[ix, ObsIdx.base_def] = pokemon.base_stats["def"] / 250.0  # Eternatus
            state[ix, ObsIdx.base_spa] = pokemon.base_stats["spa"] / 194.0  # Mega Mewtwo Y
            state[ix, ObsIdx.base_spd] = pokemon.base_stats["spd"] / 250.0  # Eternatus
            state[ix, ObsIdx.base_spe] = pokemon.base_stats["spe"] / 200.0  # Regieleki

            state[ix, ObsIdx.boost_acc] = pokemon.boosts["accuracy"] / 4.0
            state[ix, ObsIdx.boost_eva] = pokemon.boosts["evasion"] / 4.0
            state[ix, ObsIdx.boost_atk] = pokemon.boosts["atk"] / 4.0
            state[ix, ObsIdx.boost_def] = pokemon.boosts["def"] / 4.0
            state[ix, ObsIdx.boost_spa] = pokemon.boosts["spa"] / 4.0
            state[ix, ObsIdx.boost_spd] = pokemon.boosts["spd"] / 4.0
            state[ix, ObsIdx.boost_spe] = pokemon.boosts["spe"] / 4.0

            if pokemon.status is not None:
                state[ix, ObsIdx.status] = pokemon.status.value / 3.0

            all_moves = [move for move in pokemon.moves]
            if len(all_moves) > 4:
                for effect in pokemon.effects:
                    if effect.name == "DYNAMAX":
                        all_moves = all_moves[len(all_moves) - 4 :]
                        break
                else:
                    all_moves = all_moves[0:4]

            for move_ix, move in enumerate(all_moves):

                state[ix, ObsIdx.move_1_accuracy + 7 * move_ix] = pokemon.moves[move].accuracy
                state[ix, ObsIdx.move_1_base_power + 7 * move_ix] = pokemon.moves[move].accuracy
                state[ix, ObsIdx.move_1_category + 7 * move_ix] = (
                    pokemon.moves[move].category.value / 3.0
                )
                state[ix, ObsIdx.move_1_current_pp + 7 * move_ix] = (
                    pokemon.moves[move].current_pp / pokemon.moves[move].max_pp
                )
                if len(pokemon.moves[move].secondary) > 0:
                    state[ix, ObsIdx.move_1_secondary_chance + 7 * move_ix] = (
                        pokemon.moves[move].secondary[0]["chance"] / 100
                    )
                    if "status" in pokemon.moves[move].secondary[0]:
                        for value in Status:
                            if pokemon.moves[move].secondary[0]["status"] == value.name.lower():
                                state[ix, ObsIdx.move_1_secondary_status + 7 * move_ix] = (
                                    value.value / 7
                                )
                state[ix, ObsIdx.move_1_type + 7 * move_ix] = pokemon.moves[move].type.value / 19.0

        for ix, pokemon in enumerate(battle.opponent_team):
            # Divide by the number of types + 1 (for no type)
            state[ix + 6, ObsIdx.type1] = battle.opponent_team[pokemon].type_1.value / 19.0
            if battle.opponent_team[pokemon].type_2 is not None:
                state[ix + 6, ObsIdx.type2] = battle.opponent_team[pokemon].type_2.value / 19.0

            # Blissey has the maximum HP at 714
            if battle.opponent_team[pokemon].current_hp is not None:
                state[ix + 6, ObsIdx.current_hp] = battle.opponent_team[pokemon].current_hp / 714.0

            state[ix + 6, ObsIdx.hp_ratio] = battle.opponent_team[pokemon].current_hp_fraction

            state[ix + 6, ObsIdx.base_hp] = (
                battle.opponent_team[pokemon].base_stats["hp"] / 255.0
            )  # Blissey
            state[ix + 6, ObsIdx.base_atk] = (
                battle.opponent_team[pokemon].base_stats["atk"] / 190.0
            )  # Mega Mewtwo X
            state[ix + 6, ObsIdx.base_def] = (
                battle.opponent_team[pokemon].base_stats["def"] / 250.0
            )  # Eternatus
            state[ix + 6, ObsIdx.base_spa] = (
                battle.opponent_team[pokemon].base_stats["spa"] / 194.0
            )  # Mega Mewtwo Y
            state[ix + 6, ObsIdx.base_spd] = (
                battle.opponent_team[pokemon].base_stats["spd"] / 250.0
            )  # Eternatus
            state[ix + 6, ObsIdx.base_spe] = (
                battle.opponent_team[pokemon].base_stats["spe"] / 200.0
            )  # Regieleki

            state[ix + 6, ObsIdx.boost_acc] = battle.opponent_team[pokemon].boosts["accuracy"] / 4.0
            state[ix + 6, ObsIdx.boost_eva] = battle.opponent_team[pokemon].boosts["evasion"] / 4.0
            state[ix + 6, ObsIdx.boost_atk] = battle.opponent_team[pokemon].boosts["atk"] / 4.0
            state[ix + 6, ObsIdx.boost_def] = battle.opponent_team[pokemon].boosts["def"] / 4.0
            state[ix + 6, ObsIdx.boost_spa] = battle.opponent_team[pokemon].boosts["spa"] / 4.0
            state[ix + 6, ObsIdx.boost_spd] = battle.opponent_team[pokemon].boosts["spd"] / 4.0
            state[ix + 6, ObsIdx.boost_spe] = battle.opponent_team[pokemon].boosts["spe"] / 4.0

            if battle.opponent_team[pokemon].status is not None:
                state[ix + 6, ObsIdx.status] = battle.opponent_team[pokemon].status.value / 3.0

            all_moves = [move for move in battle.opponent_team[pokemon].moves]
            if len(all_moves) > 4:
                for effect in battle.opponent_team[pokemon].effects:
                    if effect.name == "DYNAMAX":
                        all_moves = all_moves[len(all_moves) - 4 :]
                        break
                else:
                    all_moves = all_moves[0:4]

            for move_ix, move in enumerate(all_moves):
                state[ix + 6, ObsIdx.move_1_accuracy + 7 * move_ix] = (
                    battle.opponent_team[pokemon].moves[move].accuracy
                )
                state[ix + 6, ObsIdx.move_1_base_power + 7 * move_ix] = (
                    battle.opponent_team[pokemon].moves[move].accuracy
                )
                state[ix + 6, ObsIdx.move_1_category + 7 * move_ix] = (
                    battle.opponent_team[pokemon].moves[move].category.value / 3.0
                )
                state[ix + 6, ObsIdx.move_1_current_pp + 7 * move_ix] = (
                    battle.opponent_team[pokemon].moves[move].current_pp
                    / battle.opponent_team[pokemon].moves[move].max_pp
                )

                if len(battle.opponent_team[pokemon].moves[move].secondary) > 0:
                    state[ix + 6, ObsIdx.move_1_secondary_chance + 7 * move_ix] = (
                        battle.opponent_team[pokemon].moves[move].secondary[0]["chance"] / 100
                    )
                    if "status" in battle.opponent_team[pokemon].moves[move].secondary[0]:
                        for value in Status:
                            if (
                                battle.opponent_team[pokemon].moves[move].secondary[0]["status"]
                                == value.name.lower()
                            ):
                                state[ix + 6, ObsIdx.move_1_secondary_status + 7 * move_ix] = (
                                    value.value / 7
                                )
                state[ix + 6, ObsIdx.move_1_type + 7 * move_ix] = (
                    battle.opponent_team[pokemon].moves[move].type.value / 19.0
                )

        return state.view(1, -1)

    def new_internals(self) -> Dict[str, torch.Tensor]:
        return {
            "hx": torch.zeros(self.lstm_hidden_size),
            "cx": torch.zeros(self.lstm_hidden_size),
        }


if __name__ == "__main__":
    test_network = LSTMNetwork(10)
    A = torch.zeros((16, 12 * 55))
    hx = torch.zeros((16, 512))
    cx = torch.zeros((16, 512))
    print(test_network(A, {"hx": hx, "cx": cx}))

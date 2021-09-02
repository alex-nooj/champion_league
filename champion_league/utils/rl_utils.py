import json
import os
import torch
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status

from champion_league.env.base.obs_idx import ObsIdx
from champion_league.utils.abilities import ABILITIES
from champion_league.utils.directory_utils import DotDict


def save_args(args: DotDict):
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    savedir = os.path.join(args.logdir, args.tag)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    with open(os.path.join(savedir, "args.json"), "w") as fp:
        json.dump(args, fp)


def load_model(args, network):
    directory_contents = os.listdir(os.path.join(args.logdir, args.tag))

    epoch_directories = [
        int(directory_name.rsplit("_")[-1])
        for directory_name in directory_contents
        if "epoch" in directory_name
    ]

    epoch_directories.sort()
    epoch_directories.reverse()

    for epoch in epoch_directories:
        try:
            checkpoint = torch.load(
                os.path.join(
                    args.logdir,
                    args.tag,
                    f"{args.tag}_epoch_{epoch}",
                    f"{args.tag}_epoch_{epoch}.pt",
                ),
                map_location=lambda storage, loc: storage,
            )
            network.load_state_dict(checkpoint)

        except:
            continue
        finally:
            break

    return network


def embed_battle(battle: Battle):
    state = torch.zeros((len(ObsIdx), 12))
    # For pokemon in battle:
    all_pokemon = [battle.active_pokemon] + battle.available_switches
    for ix, pokemon in enumerate(all_pokemon):
        # Divide by the number of types + 1 (for no type)
        state[ObsIdx.type1, ix] = pokemon.type_1.value / 19.0
        if pokemon.type_2 is not None:
            state[ObsIdx.type2, ix] = pokemon.type_2.value / 19.0

        if pokemon.ability is not None:
            state[ObsIdx.ability_bit0 : ObsIdx.ability_bit0 + 9, ix] = torch.tensor(
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
            state[ObsIdx.current_hp, ix] = pokemon.current_hp / 714.0

        state[ObsIdx.hp_ratio, ix] = pokemon.current_hp_fraction

        state[ObsIdx.base_hp, ix] = pokemon.base_stats["hp"] / 255.0  # Blissey
        state[ObsIdx.base_atk, ix] = pokemon.base_stats["atk"] / 190.0  # Mega Mewtwo X
        state[ObsIdx.base_def, ix] = pokemon.base_stats["def"] / 250.0  # Eternatus
        state[ObsIdx.base_spa, ix] = pokemon.base_stats["spa"] / 194.0  # Mega Mewtwo Y
        state[ObsIdx.base_spd, ix] = pokemon.base_stats["spd"] / 250.0  # Eternatus
        state[ObsIdx.base_spe, ix] = pokemon.base_stats["spe"] / 200.0  # Regieleki

        state[ObsIdx.boost_acc, ix] = pokemon.boosts["accuracy"] / 4.0
        state[ObsIdx.boost_eva, ix] = pokemon.boosts["evasion"] / 4.0
        state[ObsIdx.boost_atk, ix] = pokemon.boosts["atk"] / 4.0
        state[ObsIdx.boost_def, ix] = pokemon.boosts["def"] / 4.0
        state[ObsIdx.boost_spa, ix] = pokemon.boosts["spa"] / 4.0
        state[ObsIdx.boost_spd, ix] = pokemon.boosts["spd"] / 4.0
        state[ObsIdx.boost_spe, ix] = pokemon.boosts["spe"] / 4.0

        if pokemon.status is not None:
            state[ObsIdx.status, ix] = pokemon.status.value / 3.0

        all_moves = [move for move in pokemon.moves]
        if len(all_moves) > 4:
            for effect in pokemon.effects:
                if effect.name == "DYNAMAX":
                    all_moves = all_moves[len(all_moves) - 4 :]
                    break
            else:
                all_moves = all_moves[0:4]

        for move_ix, move in enumerate(all_moves):

            state[ObsIdx.move_1_accuracy + 7 * move_ix, ix] = pokemon.moves[
                move
            ].accuracy
            state[ObsIdx.move_1_base_power + 7 * move_ix, ix] = pokemon.moves[
                move
            ].accuracy
            state[ObsIdx.move_1_category + 7 * move_ix, ix] = (
                pokemon.moves[move].category.value / 3.0
            )
            state[ObsIdx.move_1_current_pp + 7 * move_ix, ix] = (
                pokemon.moves[move].current_pp / pokemon.moves[move].max_pp
            )
            if len(pokemon.moves[move].secondary) > 0:
                state[ObsIdx.move_1_secondary_chance + 7 * move_ix, ix] = (
                    pokemon.moves[move].secondary[0]["chance"] / 100
                )
                if "status" in pokemon.moves[move].secondary[0]:
                    for value in Status:
                        if (
                            pokemon.moves[move].secondary[0]["status"]
                            == value.name.lower()
                        ):
                            state[ObsIdx.move_1_secondary_status + 7 * move_ix, ix] = (
                                value.value / 7
                            )
            state[ObsIdx.move_1_type + 7 * move_ix, ix] = (
                pokemon.moves[move].type.value / 19.0
            )

    for ix, pokemon in enumerate(battle.opponent_team):
        # Divide by the number of types + 1 (for no type)
        state[ObsIdx.type1, ix + 6] = battle.opponent_team[pokemon].type_1.value / 19.0
        if battle.opponent_team[pokemon].type_2 is not None:
            state[ObsIdx.type2, ix + 6] = (
                battle.opponent_team[pokemon].type_2.value / 19.0
            )

        # Blissey has the maximum HP at 714
        if battle.opponent_team[pokemon].current_hp is not None:
            state[ObsIdx.current_hp, ix + 6] = (
                battle.opponent_team[pokemon].current_hp / 714.0
            )

        state[ObsIdx.hp_ratio, ix + 6] = battle.opponent_team[
            pokemon
        ].current_hp_fraction

        state[ObsIdx.base_hp, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["hp"] / 255.0
        )  # Blissey
        state[ObsIdx.base_atk, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["atk"] / 190.0
        )  # Mega Mewtwo X
        state[ObsIdx.base_def, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["def"] / 250.0
        )  # Eternatus
        state[ObsIdx.base_spa, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["spa"] / 194.0
        )  # Mega Mewtwo Y
        state[ObsIdx.base_spd, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["spd"] / 250.0
        )  # Eternatus
        state[ObsIdx.base_spe, ix + 6] = (
            battle.opponent_team[pokemon].base_stats["spe"] / 200.0
        )  # Regieleki

        state[ObsIdx.boost_acc, ix + 6] = (
            battle.opponent_team[pokemon].boosts["accuracy"] / 4.0
        )
        state[ObsIdx.boost_eva, ix + 6] = (
            battle.opponent_team[pokemon].boosts["evasion"] / 4.0
        )
        state[ObsIdx.boost_atk, ix + 6] = (
            battle.opponent_team[pokemon].boosts["atk"] / 4.0
        )
        state[ObsIdx.boost_def, ix + 6] = (
            battle.opponent_team[pokemon].boosts["def"] / 4.0
        )
        state[ObsIdx.boost_spa, ix + 6] = (
            battle.opponent_team[pokemon].boosts["spa"] / 4.0
        )
        state[ObsIdx.boost_spd, ix + 6] = (
            battle.opponent_team[pokemon].boosts["spd"] / 4.0
        )
        state[ObsIdx.boost_spe, ix + 6] = (
            battle.opponent_team[pokemon].boosts["spe"] / 4.0
        )

        if battle.opponent_team[pokemon].status is not None:
            state[ObsIdx.status, ix + 6] = (
                battle.opponent_team[pokemon].status.value / 3.0
            )

        all_moves = [move for move in battle.opponent_team[pokemon].moves]
        if len(all_moves) > 4:
            for effect in battle.opponent_team[pokemon].effects:
                if effect.name == "DYNAMAX":
                    all_moves = all_moves[len(all_moves) - 4 :]
                    break
            else:
                all_moves = all_moves[0:4]

        for move_ix, move in enumerate(all_moves):
            state[ObsIdx.move_1_accuracy + 7 * move_ix, ix] = (
                battle.opponent_team[pokemon].moves[move].accuracy
            )
            state[ObsIdx.move_1_base_power + 7 * move_ix, ix] = (
                battle.opponent_team[pokemon].moves[move].accuracy
            )
            state[ObsIdx.move_1_category + 7 * move_ix, ix] = (
                battle.opponent_team[pokemon].moves[move].category.value / 3.0
            )
            state[ObsIdx.move_1_current_pp + 7 * move_ix, ix] = (
                battle.opponent_team[pokemon].moves[move].current_pp
                / battle.opponent_team[pokemon].moves[move].max_pp
            )

            if len(battle.opponent_team[pokemon].moves[move].secondary) > 0:
                state[ObsIdx.move_1_secondary_chance + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].secondary[0]["chance"]
                    / 100
                )
                if "status" in battle.opponent_team[pokemon].moves[move].secondary[0]:
                    for value in Status:
                        if (
                            battle.opponent_team[pokemon]
                            .moves[move]
                            .secondary[0]["status"]
                            == value.name.lower()
                        ):
                            state[ObsIdx.move_1_secondary_status + 7 * move_ix, ix] = (
                                value.value / 7
                            )
            state[ObsIdx.move_1_type + 7 * move_ix, ix] = (
                battle.opponent_team[pokemon].moves[move].type.value / 19.0
            )

    return state

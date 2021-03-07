import torch
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.weather import Weather
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status

from champion_league.env.base.obs_idx import ObsIdx
from champion_league.utils.abilities import ABILITIES

POKEMON_LEN = 27
MOVE_LEN = 40


def encode_battle(battle: AbstractBattle) -> torch.Tensor:
    encoded_battle = torch.zeros(6, (POKEMON_LEN + 4*MOVE_LEN)).float()

    for pokemon_ix, pokemon in enumerate(battle.team):
        encoded_battle[pokemon_ix, 0:POKEMON_LEN] = encode_pokemon(battle.team[pokemon])
        for move_ix, move in enumerate(battle.team[pokemon].moves):
            if move_ix == 4:
                break
            new_move = encode_move(battle.team[pokemon].moves[move])
            encoded_battle[pokemon_ix, POKEMON_LEN + move_ix * MOVE_LEN: POKEMON_LEN + (move_ix + 1) * MOVE_LEN] = new_move
    return encoded_battle

def encode_pokemon(pokemon: Pokemon) -> torch.Tensor:
    encoded_pokemon = torch.zeros(27).float()

    # Divide by the number of types + 1 (for no type)
    encoded_pokemon[ObsIdx.type1] = pokemon.type_1.value / 19.0
    if pokemon.type_2 is not None:
        encoded_pokemon[ObsIdx.type2] = pokemon.type_2.value / 19.0
    else:
        encoded_pokemon[ObsIdx.type2] = 1.0
    # TODO: Abilities need to be binary
    encoded_pokemon[ObsIdx.ability_bit0:ObsIdx.ability_bit0+9] = torch.tensor(ABILITIES[
        pokemon.ability.lower().replace(" ", "").replace("-", "")])

    # Blissey has the maximum HP at 714
    if pokemon.current_hp is None:
        encoded_pokemon[ObsIdx.current_hp] = 0.0
    else:
        encoded_pokemon[ObsIdx.current_hp] = pokemon.current_hp / 1428.0

    encoded_pokemon[ObsIdx.hp_ratio] = pokemon.current_hp_fraction

    encoded_pokemon[ObsIdx.base_hp] = pokemon.base_stats["hp"] / 255.0  # Blissey
    encoded_pokemon[ObsIdx.base_atk] = pokemon.base_stats["atk"] / 190.0  # Mega Mewtwo X
    encoded_pokemon[ObsIdx.base_def] = pokemon.base_stats["def"] / 250.0  # Eternatus
    encoded_pokemon[ObsIdx.base_spa] = pokemon.base_stats["spa"] / 194.0  # Mega Mewtwo Y
    encoded_pokemon[ObsIdx.base_spd] = pokemon.base_stats["spd"] / 250.0  # Eternatus
    encoded_pokemon[ObsIdx.base_spe] = pokemon.base_stats["spe"] / 200.0  # Regieleki

    encoded_pokemon[ObsIdx.boost_acc] = pokemon.boosts["accuracy"] / 6.0
    encoded_pokemon[ObsIdx.boost_eva] = pokemon.boosts["evasion"] / 6.0
    encoded_pokemon[ObsIdx.boost_atk] = pokemon.boosts["atk"] / 6.0
    encoded_pokemon[ObsIdx.boost_def] = pokemon.boosts["def"] / 6.0
    encoded_pokemon[ObsIdx.boost_spa] = pokemon.boosts["spa"] / 6.0
    encoded_pokemon[ObsIdx.boost_spd] = pokemon.boosts["spd"] / 6.0
    encoded_pokemon[ObsIdx.boost_spe] = pokemon.boosts["spe"] / 6.0

    if pokemon.status is not None:
        encoded_pokemon[ObsIdx.status] = pokemon.status.value / (len(Status) + 1)

    return encoded_pokemon


def encode_move(move: Move) -> torch.Tensor:
    encoded_move = torch.zeros(40).float()

    encoded_move[0] = move.accuracy
    encoded_move[1] = move.base_power / 250 # Eruption
    if move.boosts is not None:
        if "atk" in move.boosts:
            encoded_move[2] = move.boosts["atk"] / 4.0
        if "def" in move.boosts:
            encoded_move[3] = move.boosts["def"] / 4.0
        if "spa" in move.boosts:
            encoded_move[4] = move.boosts["spa"] / 4.0
        if "spd" in move.boosts:
            encoded_move[5] = move.boosts["spd"] / 4.0
        if "spe" in move.boosts:
            encoded_move[6] = move.boosts["spe"] / 4.0
    encoded_move[7] = float(move.breaks_protect)
    encoded_move[8] = float(move.can_z_move)

    # TODO: Check this
    encoded_move[9] = move.category.value / 3.0

    encoded_move[10] = move.crit_ratio / 6.0
    encoded_move[11] = move.current_pp / move.max_pp
    encoded_move[13] = move.defensive_category.value / 3.0
    encoded_move[14] = move.drain
    encoded_move[15] = move.expected_hits / 6.0
    encoded_move[16] = float(move.force_switch)
    encoded_move[17] = move.heal
    if move.ignore_ability:
        encoded_move[18] = 1.0
    if move.ignore_defensive:
        encoded_move[19] = 1.0
    if move.ignore_evasion:
        encoded_move[20] = 1.0
    if move.ignore_immunity:
        encoded_move[21] = 1.0
    if move.is_empty:
        encoded_move[22] = 1.0
    if move.is_z:
        encoded_move[23] = 1.0
    encoded_move[24] = move.priority / 7.0
    encoded_move[25] = move.recoil
    if move.self_boost is not None:
        if "atk" in move.self_boost:
            encoded_move[26] = move.self_boost["atk"] / 4.0
        if "def" in move.self_boost:
            encoded_move[27] = move.self_boost["def"] / 4.0
        if "spa" in move.self_boost:
            encoded_move[28] = move.self_boost["spa"] / 4.0
        if "spd" in move.self_boost:
            encoded_move[29] = move.self_boost["spd"] / 4.0
        if "spe" in move.self_boost:
            encoded_move[30] = move.self_boost["spe"] / 4.0
    if move.self_destruct is not None:
        encoded_move[31] = 1.0
    encoded_move[32] = move.self_switch
    encoded_move[33] = move.sleep_usable
    encoded_move[34] = move.stalling_move
    if move.status is not None:
        encoded_move[35] = move.status.value / (len(Status) + 1)
    encoded_move[36] = move.steals_boosts
    encoded_move[37] = move.thaws_target
    encoded_move[38] = move.type.value / 19
    if move.weather is not None:
        encoded_move[39] = move.weather.value / (len(Weather) + 1)

    return encoded_move

import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

Pokemon = Dict[str, Union[str, Dict[str, int], List[str], int]]

CURR_PATH = pathlib.Path(__file__)
for parent in CURR_PATH.parents:
    if str(parent).endswith("champion_league"):
        TEAM_PATH = parent / "teams"
        break


def generate_random_team(battle_format: Optional[str] = "gen8ou") -> str:
    path_to_teams = TEAM_PATH / battle_format
    pokemon_pool = [p for p in path_to_teams.iterdir()]
    pokemon_team = np.random.choice(pokemon_pool, size=6, replace=False)
    team = ""
    for pokemon in pokemon_team:
        moveset = np.random.choice([m for m in pokemon.iterdir()])
        with open(moveset) as f:
            team += f.read()
        team += "\n\n"
    return team

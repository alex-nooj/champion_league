import json

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType

from champion_league.utils.random_team_generator import TEAM_PATH

path_to_teams = TEAM_PATH / "gen8ou"

pokemon_pool = [p for p in path_to_teams.iterdir()]

pokemon = []

for mon in pokemon_pool:
    for moveset in mon.iterdir():
        with open(moveset) as f:
            pokemon.append(Pokemon(species=f.read().rsplit(" ")[0]))

with open(
    "/home/anewgent/Documents/poke-env/src/poke_env/data/typeChart.json", "r"
) as fp:
    types = [PokemonType.from_name(v["name"]) for v in json.load(fp)]

scores = {}
for mon in pokemon:
    if mon.species in scores:
        continue
    scores[mon.species] = {
        move_type.name: mon.damage_multiplier(move_type) for move_type in types
    }

unresisted = [t.name for t in types]

print()

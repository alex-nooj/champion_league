import json
from poke_env.data import POKEDEX

abilities = {}
count = 0

if __name__ == "__main__":
    for key in POKEDEX:
        for ability in POKEDEX[key]["abilities"]:
            if POKEDEX[key]["abilities"][ability].lower().replace(" ", "") not in abilities:
                binary_count = [int(i) for i in bin(count)[2:]]

                abilities[
                    POKEDEX[key]["abilities"][ability]
                    .lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("(", "")
                    .replace(")", "")
                ] = [0] * (9 - len(binary_count)) + binary_count
                count += 1

print(abilities)

with open("/home/alex/Documents/champion_league/champion_league/utils/abilities.json", "w",) as fp:
    json.dump(abilities, fp, indent=4)

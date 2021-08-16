import json

from poke_env.environment.status import Status

move_keys = []

if __name__ == "__main__":
    # with open("/home/anewgent/anaconda3/envs/sc2/lib/python3.8/site-packages/poke_env/data/moves.json", "r") as fp:
    #     moves = json.load(fp)

    for value in Status:
        print(value.name)
    # for move in moves:
    #     for key in moves[move]:
    #         if key not in move_keys:
    #             print(key)
    #             move_keys.append(key)

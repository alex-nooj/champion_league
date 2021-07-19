import random


def agent(env):
    moves = env.current_battle["actions"][1]
    return random.choice(moves)

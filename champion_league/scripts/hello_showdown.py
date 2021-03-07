# -*- coding: utf-8 -*-
import numpy as np
from adept.scripts.local import parse_args, main

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
import adept


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    ids = ["SimpleRLPlayer"]

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

np.random.seed(0)


# This is the function that will be used to train the dqn
def agent_training(player, args):
    main(args)
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


if __name__ == "__main__":
    adept.register_env(PokeEnv)

    args = parse_args()
    args.agent = "ActorCritic"
    args.tag = "First_Run"
    args.logdir = "/home/alex/Documents/pokemon_agents/"
    args.env = "PokeEnv"
    args.battle_format = "gen8randombattle"
    args.player_configuration = None
    env_player = PokeEnv(SimpleRLPlayer(battle_format="gen8randombattle"))

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # Output dimension
    n_action = len(env_player.action_space)

    # Training
    env_player.play_against(
        env_algorithm=agent_training,
        opponent=opponent,
        env_algorithm_kwargs={"args": args},
    )

    # # Evaluation
    # print("Results against random player:")
    # env_player.play_against(
    #     env_algorithm=dqn_evaluation,
    #     opponent=opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    # )
    #
    # print("\nResults against max player:")
    # env_player.play_against(
    #     env_algorithm=dqn_evaluation,
    #     opponent=second_opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    # )
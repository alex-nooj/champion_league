"""
Train a DQN Agent to play Gen8Random battles

Usage:
    simple_dqn [options]

General Options:
    --nb_train_episodes <int>       How many episodes to run training for [default: 2000]
    --nb_eval_episodes <int>        How many episodes to evalutate on [default: 100]
    --nb_epochs <int>               How many epochs to run [default: 1000]
    --batch_size <int>              Batch size [default: 256]
    --tag <str>                     Name of the network [default: None]
    --device <int>                  GPU to train on [default: 0]
    --logdir <str>                  Where to save the model [default: /home/anewgent/Documents/pokemon_agents/dqn_start/]
    --algorithm <str>               Which algorithm to use [default: DQN]
    --opponent_logdir <str>         Where to load the opponenet from [default: /home/anewgent/Documents/pokemon_agents/dqn_start/]
    --opponent_tag <str>            Name of the opponent [default: sampling_actions]
    --opponent_gpu <int>            GPU to load the opponent onto [default: 0]
"""
import time
from typing import Callable
from typing import Tuple

import numpy as np
import torch
from champion_league.utils.directory_utils import DotDictfrom poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.dqn.agent import DQNAgent
from champion_league.agent.dqn.utils import sampling_policy
from champion_league.agent.scripted.max_damage_player import MaxDamagePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.utils.parse_args import parse_args

NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100
TARGET_UPDATE = 10
START_TIME = time.time()


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
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


np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(
    player,
    agent,
    args,
    opponent,
    selection_policy: Callable,
):
    print(opponent.name)
    for i_episode in range(args.nb_train_episodes):
        state = player.reset()

        done = False
        nb_steps = 0
        start_time = time.time()
        while not done:
            state = state.to(args.device).double()
            action = agent.choose_move(state.view(1, -1))

            next_state, reward, done, info = player.step(action.item())
            reward = torch.tensor([reward], device=args.device)

            agent.memory.push(state, action, next_state.double(), reward)
            state = next_state

            agent.learn_step()
            nb_steps += 1
        end_time = time.time()
        print(
            f"{i_episode}: {nb_steps/(end_time - start_time): 0.3f} steps/sec, REWARD: {int(reward[0])}"
        )

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    player.complete_current_battle()


def dqn_evaluation(player, policy_net, args, epoch, opponent):
    # Reset battle statistics
    print(f"STARTUP TIME: {time.time() - START_TIME}")

    player.reset_battles()
    for i_episode in range(args.nb_eval_episodes):
        state = player.reset()
        done = False
        while not done:
            print(opponent.name)
            # state = torch.from_numpy(state)
            state = torch.reshape(state, (1, -1)).to(args.device).double()
            action = policy_net(state.float()).max(1)[1].view(1, 1)
            state, reward, done, info = player.step(action.item())

    print(
        "Epoch %d: %0.3f win rate"
        % (epoch, player.n_won_battles / args.nb_eval_episodes)
    )


def main(args: DotDict, opponent_args: DotDict):
    env_player = RLPlayer(
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration(args.tag, "test"),
    )

    # Output dimension
    args.nb_actions = len(env_player.action_space)

    second_opponent = MaxDamagePlayer(
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration("max_damage", "test"),
    )

    agent = DQNAgent(args, PlayerConfiguration("whatever", "test"))

    for epoch in range(args.nb_epochs):
        env_player.play_against(
            env_algorithm=dqn_evaluation,
            opponent=second_opponent,
            env_algorithm_kwargs={
                "policy_net": agent.policy_net.eval().to(args.device),
                "args": args,
                "epoch": epoch,
                "opponent": second_opponent,
            },
        )

        env_player.play_against(
            env_algorithm=dqn_training,
            opponent=second_opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "args": args,
                "selection_policy": sampling_policy,
                "opponent": second_opponent,
            },
        )
        agent.save_model(agent.network, epoch)


if __name__ == "__main__":
    args, opponent_args = parse_args()
    main(args, opponent_args)

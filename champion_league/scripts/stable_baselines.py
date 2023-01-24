import asyncio

import torch
from poke_env.player import MaxBasePowerPlayer
from poke_env.player import RandomPlayer
from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from champion_league.callbacks.curriculum_callback import CurriculumCallback
from champion_league.callbacks.save_callback import SaveCallback
from champion_league.callbacks.win_rates_callback import WinRatesCallback
from champion_league.env.pokemon_env import PokemonEnv
from champion_league.preprocessor.simple_preprocessor import SimplePreprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.agent_team_builder import AgentTeamBuilder
from champion_league.utils.directory_utils import PokePath


async def main():
    # Create one environment for training and one for evaluation
    opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=AgentTeamBuilder())
    preprocessor = SimplePreprocessor()
    reward_scheme = RewardScheme(
        rules={
            "OpponentBinaryFaints": 1 / 6,
            "AlliedBinaryFaints": 1 / 6,
            "VictoryRule": 1.0,
            "LossRule": 1.0,
        }
    )

    logdir = "/home/anewgent/Projects/pokemon/kanto_league"
    tag = "Brock"
    epoch_len = 100_000

    poke_path = PokePath(logdir, tag)
    poke_path.agent.mkdir(parents=True, exist_ok=True)
    team_builder = AgentTeamBuilder(
        agent_path=poke_path.agent,
    )
    train_env = PokemonEnv(
        opponent=opponent,
        preprocessor=preprocessor,
        reward_scheme=reward_scheme,
        battle_format="gen8ou",
        start_challenging=True,
        team=team_builder,
    )

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 32, 16])
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=str(poke_path.agent),
        policy_kwargs=policy_kwargs,
    )

    callback = CallbackList(
        [
            SaveCallback(league_path=poke_path, epoch_len=epoch_len, verbose=1),
            CurriculumCallback(
                opponents=[
                    RandomPlayer(battle_format="gen8ou", team=AgentTeamBuilder()),
                    MaxBasePowerPlayer(battle_format="gen8ou", team=AgentTeamBuilder()),
                    SimpleHeuristicsPlayer(
                        battle_format="gen8ou", team=AgentTeamBuilder()
                    ),
                ],
                verbose=2,
            ),
            WinRatesCallback(),
        ]
    )
    model.learn(
        total_timesteps=100_000_000,
        reset_num_timesteps=False,
        tb_log_name="agent_play",
        progress_bar=True,
        callback=callback,
    )
    train_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

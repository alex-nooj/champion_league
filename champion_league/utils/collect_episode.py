from typing import Optional

from champion_league.agent.ppo import PPOAgent
from champion_league.env import RLPlayer
from champion_league.utils.replay import Episode
from champion_league.utils.step_counter import StepCounter


def collect_episode(
    player: RLPlayer, agent: PPOAgent, step_counter: Optional[StepCounter] = None
) -> Episode:
    internals = agent.network.reset(device=agent.device)
    observation = player.reset()
    episode = Episode()
    done = False

    while not done:
        action, log_prob, value, new_internals = agent.sample_action(
            observation, internals
        )

        new_observation, reward, done, info = player.step(action)
        if step_counter is not None:
            step_counter()
        episode.append(
            observation=observation,
            internals=internals,
            action=action,
            reward=reward,
            value=value,
            log_probability=log_prob,
            reward_scale=player.reward_scheme.max,
        )

        observation = new_observation
        internals = new_internals

    episode.end_episode(last_value=0)

    return episode

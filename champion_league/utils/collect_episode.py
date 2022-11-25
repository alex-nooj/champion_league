from typing import Optional

from champion_league.agent.base.base_agent import Agent
from champion_league.env import RLPlayer
from champion_league.utils.replay import Episode
from champion_league.utils.step_counter import StepCounter


def collect_episode(
    player: RLPlayer, agent: Agent, step_counter: Optional[StepCounter] = None
) -> Episode:
    observation = player.reset()
    episode = Episode()
    done = False
    ep_reward = []
    while not done:
        action, log_prob, value = agent.sample_action(observation)

        new_observation, reward, done, info = player.step(action)
        if step_counter is not None:
            step_counter()

        for k, v in reward.items():
            agent.log_scalar(f"Rewards/{k}", v)
        total_reward = sum([v for v in reward.values()])
        ep_reward.append(total_reward)
        episode.append(
            observation=observation,
            action=action,
            reward=total_reward,
            value=value,
            log_probability=log_prob,
            reward_scale=player.reward_scheme.max,
        )

        observation = new_observation
    agent.log_scalar(f"Rewards/Total", sum(ep_reward))
    episode.end_episode(last_value=0)

    return episode

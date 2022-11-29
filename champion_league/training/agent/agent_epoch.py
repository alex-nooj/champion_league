import numpy as np

from champion_league.agent.base.base_agent import Agent
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.step_counter import StepCounter


def agent_epoch(
    player: RLPlayer,
    agent: Agent,
    opponent: LeaguePlayer,
    epoch_len: int,
    step_counter: StepCounter,
    epoch: int,
) -> None:
    start_step = step_counter.steps

    while step_counter.steps - start_step < epoch_len or len(agent.replay_buffer) != 0:
        opponent.change_agent(None, None)
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)
        agent.update_winrates(opponent.tag, int(episode.rewards[-1] > 0))
        agent.log_scalar(
            f"Agent Play/{opponent.tag}", np.mean(agent.win_rates[opponent.tag])
        )
        agent.replay_buffer.add_episode(episode)
        if agent.learn_step(epoch):
            agent.save_model(epoch, agent.network, player.preprocessor, player.team)

import numpy as np

from champion_league.agent.base.base_agent import Agent
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.training.league.league_skill_tracker import LeagueSkillTracker
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.step_counter import StepCounter


def league_epoch(
    player: RLPlayer,
    agent: Agent,
    opponent: LeaguePlayer,
    skill_tracker: LeagueSkillTracker,
    epoch_len: int,
    step_counter: StepCounter,
    epoch: int,
) -> None:
    """Runs one epoch of league-style training. This function is meant to be passed into the
    player's `play_against()` function.

    Args:
        player: The player that is actually running the game. This acts as our environment.
        agent: Handles the sampling of moves and also performing the learn step for the agent.
        opponent: The agent that is playing against the training agent. This agent handles scheme
            selection, but its actual action selection is being done in a separate thread.
        skill_tracker: This object tracks the trueskill of the agent, as well as all the agents in
            the league.
        epoch_len: How many steps are in an epoch.
        step_counter: Tracks the total number of steps across each epoch.
        epoch: The current training epoch.
    """
    start_step = step_counter.steps
    opponent_name = opponent.change_agent(
        skill_tracker.agent_skill,
        skill_tracker.skill_ratings,
    )

    while step_counter.steps - start_step < epoch_len or len(agent.replay_buffer) != 0:
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)

        agent.log_scalar(
            "Agent Outputs/Average Episode Reward",
            float(np.sum(episode.rewards)),
        )

        if opponent.tag.rsplit("_")[0] != agent.tag:
            agent.update_winrates(opponent_name, int(episode.rewards[-1] > 0))
            agent.log_scalar(
                f"League Training/{opponent_name}",
                float(np.mean(agent.win_rates[opponent_name])),
            )
            skill_tracker.update(episode.rewards[-1] > 0, opponent_name)
            for k, v in skill_tracker.skill.items():
                agent.log_scalar(f"True Skill/{k}", v)
        else:
            tag = "self" if int(opponent.tag.rsplit("_")[-1]) == epoch else "prev_self"
            agent.update_winrates(tag, int(episode.rewards[-1] > 0))
            agent.log_scalar(
                f"League Training/{tag}",
                float(np.mean(agent.win_rates[tag])),
            )
        opponent_name = opponent.change_agent(
            skill_tracker.agent_skill,
            skill_tracker.skill_ratings,
        )
        agent.replay_buffer.add_episode(episode)
        if agent.learn_step(epoch):
            agent.save_model(epoch, agent.network, player.preprocessor, player.team)

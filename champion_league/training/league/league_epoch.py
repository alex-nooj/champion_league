import numpy as np

from champion_league.agent.base.base_agent import Agent
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.matchmaking.league_skill_tracker import LeagueSkillTracker
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.step_counter import StepCounter


def flush_teams(player: RLPlayer):
    """Runs a battle just taking the first available action.

    Because PokeEnv is a little odd, changing teams between battles is difficult. If we change teams
    after `done` is True, then we need to run another battle before our team change is reflected on
    the server.

    Args:
        player: The player that is actually running the game. This acts as our environment.
    """
    _ = player.reset()
    done = False
    while not done:
        _, _, done, _ = player.step(0)


def league_epoch(
    player: RLPlayer,
    agent: Agent,
    opponent: LeaguePlayer,
    matchmaker: MatchMaker,
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
        matchmaker: This handles the matchmaking process for the agent. It will take in the
            Trueskill values for all the agents to determine a match-up where each agent has an even
            chance at winning.
        skill_tracker: This object tracks the trueskill of the agent, as well as all of agents in
            the league.
        epoch_len: How many steps are in an epoch.
        step_counter: Tracks the total number of steps across each epoch.
        epoch: The current training epoch.
    """
    start_step = step_counter.steps
    opponent_name = matchmaker.choose_match(
        skill_tracker.agent_skill,
        skill_tracker.skill_ratings,
    )

    _ = opponent.change_agent(opponent_name)

    while step_counter.steps - start_step < epoch_len or len(agent.replay_buffer) != 0:
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)
        agent.log_scalar(
            "Agent Outputs/Average Episode Reward",
            float(np.sum(episode.rewards)),
        )

        agent.log_scalar(
            "Agent Outputs/Average Probabilities",
            float(np.mean([np.exp(lp) for lp in episode.log_probabilities])),
        )

        agent.update_winrates(opponent.tag, int(episode.rewards[-1] > 0))

        agent.log_scalar(
            f"League Training/{opponent.tag}",
            float(np.mean(agent.win_rates[opponent.tag])),
        )

        if opponent.tag != "self":
            skill_tracker.update(episode.rewards[-1] > 0, opponent.tag)
            for k, v in skill_tracker.skill.items():
                agent.log_scalar(f"True Skill/{k}", v)

        opponent_name = matchmaker.choose_match(
            skill_tracker.agent_skill,
            skill_tracker.skill_ratings,
        )
        _ = opponent.change_agent(opponent_name)
        if opponent.mode == "ml":
            flush_teams(player)
        agent.replay_buffer.add_episode(episode)

        agent.learn_step(epoch)

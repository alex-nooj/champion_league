# Step 1: Load all of the trueskills
# Step 2: Have each agent play against all of the other agents
# Step 3: Save the new trueskills
# Step 4: Print the new trueskills
import asyncio
import os
from typing import List

from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from tqdm import tqdm

from champion_league.env import OpponentPlayer
from champion_league.matchmaking.skill_tracker import MultiSkillTracker
from champion_league.utils.parse_args import parse_args
from champion_league.utils.progress_bar import centered


async def match(player1: Player, player2: Player, nb_battles: int):
    await player1.battle_against(player2, n_battles=nb_battles)


def print_trueskills(
    agent_names: List[str], agent_skills: List[float], agent_wins: List[int]
):
    name_column_size = max([len(name) for name in agent_names]) + 2
    skill_column_size = len(
        max([centered(f"{skills:0.3f}") for skills in agent_skills])
    )

    divider = (
        "+"
        + "-" * name_column_size
        + "+"
        + "-" * skill_column_size
        + "+"
        + "-" * skill_column_size
        + "+"
    )
    print(divider)
    print(
        "|"
        + centered("Name", desired_len=name_column_size)
        + "|"
        + centered("Skill", desired_len=skill_column_size)
        + "|"
        + centered("Win", desired_len=skill_column_size)
        + "|"
    )
    print(divider)
    for name, skill, win in zip(agent_names, agent_skills, agent_wins):
        print(
            "|"
            + centered(name, desired_len=name_column_size)
            + "|"
            + centered(f"{skill:0.3f}", desired_len=skill_column_size)
            + "|"
            + centered(f"{win:3d}", desired_len=skill_column_size)
            + "|"
        )
    print(divider)


def main(league_dir: str, games_per_agent: int, p1_device: int, p2_device: int):
    usernames = {}
    agent_tags = os.listdir(league_dir)
    agent_tags.sort()

    skill_tracker = MultiSkillTracker(
        agent_paths=[os.path.join(league_dir, agent_name) for agent_name in agent_tags],
    )

    agent_wins = {k: 0 for k in agent_tags}

    for ix, agent in enumerate(agent_tags[:-1]):
        if agent in usernames:
            usernames[agent] += 1
            username = f"{agent}_{usernames[agent]}"
        else:
            username = agent
            usernames[agent] = 0

        current_player = OpponentPlayer.from_path(
            path=os.path.join(league_dir, agent),
            device=p1_device,
            max_concurrent_battles=games_per_agent,
            player_configuration=PlayerConfiguration(username, "none"),
        )

        for opponent in tqdm(agent_tags[ix + 1 :]):
            if opponent in usernames:
                usernames[opponent] += 1
                username = f"{opponent}_{usernames[opponent]}"
            else:
                username = opponent
                usernames[opponent] = 0

            opponent_player = OpponentPlayer.from_path(
                path=os.path.join(league_dir, opponent),
                device=p2_device,
                max_concurrent_battles=games_per_agent,
                player_configuration=PlayerConfiguration(username, "none"),
            )

            asyncio.get_event_loop().run_until_complete(
                match(current_player, opponent_player, games_per_agent)
            )

            for victory in current_player.battle_history:
                if victory:
                    skill_tracker.update(agent, opponent)
                else:
                    skill_tracker.update(opponent, agent)
            agent_wins[agent] += current_player.n_won_battles
            agent_wins[opponent] += current_player.n_lost_battles
            current_player.reset_battles()

    print_trueskills(
        agent_names=agent_tags,
        agent_skills=[
            skill_tracker.agent_trueskill(v)["trueskill"] for v in agent_tags
        ],
        agent_wins=[agent_wins[v] for v in agent_tags],
    )

    skill_tracker.save_trueskills()


if __name__ == "__main__":
    main(**parse_args())

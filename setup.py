from setuptools import setup

setup(
    name="champion_league",
    version="0.0.1",
    packages=[
        "champion_league",
        "champion_league.agent",
        "champion_league.agent.dqn",
        "champion_league.agent.ppo",
        "champion_league.agent.scripted",
        "champion_league.utils",
        "champion_league.config",
        "champion_league.network",
        "champion_league.scripts",
        "champion_league.preprocessors",
    ],
    url="https://github.com/alex-nooj/champion_league",
    license="GNU",
    author="anewgent",
    author_email="",
    description="Reinforcement Learning for Pokemon Battling",
)

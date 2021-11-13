from setuptools import setup

setup(
    name="champion_league",
    version="0.1",
    packages=[
        "champion_league",
        "champion_league.env",
        "champion_league.agent",
        "champion_league.agent.ppo",
        "champion_league.agent.scripted",
        "champion_league.utils",
        "champion_league.config",
        "champion_league.network",
        "champion_league.scripts",
        "champion_league.preprocessors",
        "champion_league.preprocessors.modules",
    ],
    url="https://github.com/alex-nooj/champion_league",
    license="",
    author="anewgent",
    author_email="",
    description="Repo for training a neural network to be the very best, like no one ever was.",
)

from pathlib import Path
from typing import Dict
from typing import Optional

import trueskill
from omegaconf import OmegaConf

from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.poke_path import PokePath


class LeagueSkillTracker:
    def __init__(
        self,
        league_path: PokePath,
        resume: Optional[bool] = False,
    ):
        """Helper class for tracking the trueskill of an agent and the league agents.

        Parameters
        ----------
        league_path: PokePath
            Path to where the entirety of the league is being stored.
        resume: Optional[bool]
            Whether or not to load a previous trueskill rating for the training agent.
        """
        self.league_path = league_path
        self.default_mu = 25
        self.default_sigma = 8.333

        agent_mu = None
        agent_sigma = None
        if resume:
            try:
                epoch = get_most_recent_epoch(self.league_path.agent)
                skill_file = (
                    get_save_dir(self.league_path.agent, epoch) / "trueskill.yaml"
                )
                temp = OmegaConf.to_container(OmegaConf.load(skill_file))
                agent_mu = temp["mu"]
                agent_sigma = temp["sigma"]
            except (ValueError, FileNotFoundError):
                pass

        self.agent_skill = trueskill.Rating(
            mu=self.default_mu if agent_mu is None else agent_mu,
            sigma=self.default_sigma if agent_sigma is None else agent_sigma,
        )

        self.skill_ratings = self.load_league_skill()

    def load_league_skill(self) -> Dict[str, trueskill.Rating]:
        """Loads the skill ratings of all of the agents from memory.

        Returns
        -------
        Dict[str, trueskill.Rating]
            Stores the trueskill ratings for each agent, with the agent names for keys.
        """
        return {
            agent.stem: self._load_agent_skill(agent)
            for agent in self.league_path.league.iterdir()
        }

    def _load_agent_skill(self, agent_path: Path) -> trueskill.Rating:
        """Checks for the trueskill.json file for the agent specified by tag. If it exists, this
        function loads the file and returns the trueskill. Otherwise, returns the default trueskill.

        Parameters
        ----------
        agent_path: str
            The name of the agent whose trueskill we want to load.

        Returns
        -------
        trueskill.Rating
            The trueskill rating of the agent specified by tag.
        """
        trueskill_file = agent_path / "trueskill.yaml"
        if trueskill_file.is_file():
            agent_skill = OmegaConf.to_container(OmegaConf.load(trueskill_file))
        else:
            agent_skill = {"mu": self.default_mu, "sigma": self.default_sigma}
            OmegaConf.save(agent_skill, trueskill_file)

        return trueskill.Rating(**agent_skill)

    def _save_league_skill(self):
        """Saves the trueskills of all the league agents."""
        for agent, skill in self.skill_ratings.items():
            OmegaConf.save(
                {
                    "mu": skill.mu,
                    "sigma": skill.sigma,
                },
                self.league_path.league / agent / "trueskill.yaml",
            )

    def _save_agent_skill(self, epoch: int) -> None:
        """Saves the trueskill of the training agent.

        Parameters
        ----------
        epoch: int
            The current epoch of training.

        Returns
        -------
        None
        """
        trueskill_file = get_save_dir(self.league_path.agent, epoch) / "trueskill.yaml"

        OmegaConf.save(
            {
                "mu": self.agent_skill.mu,
                "sigma": self.agent_skill.sigma,
            },
            trueskill_file,
        )

    def save_skill_ratings(self, epoch: int) -> None:
        """Saves all of the trueskill values for every agent.

        Parameters
        ----------
        epoch: int
            The current epoch of training.

        Returns
        -------
        None
        """
        self._save_league_skill()
        self._save_agent_skill(epoch)

    def update(self, agent_won: bool, opponent: str) -> None:
        """Updates the trueskill of the training agent and the opponent that was just played.

        Parameters
        ----------
        agent_won: bool
            Whether the training agent won.
        opponent: str
            The name of the opponent the agent just played.

        Returns
        -------
        None
        """
        if (self.league_path.agent / opponent).is_dir():
            opponent = "self"

        try:
            _ = self.skill_ratings[opponent]
        except KeyError:
            self.skill_ratings[opponent] = self._load_agent_skill(
                self.league_path.league / opponent
            )
        finally:
            if agent_won:
                self.agent_skill, self.skill_ratings[opponent] = trueskill.rate_1vs1(
                    self.agent_skill,
                    self.skill_ratings[opponent],
                )
            else:
                self.skill_ratings[opponent], self.agent_skill = trueskill.rate_1vs1(
                    self.skill_ratings[opponent],
                    self.agent_skill,
                )

    @property
    def skill(self) -> Dict[str, float]:
        """The current skill of the training agent.

        Returns
        -------
        Dict[str, float]
            The mean and variance for the agent's trueskill rating.
        """
        return {
            "mu": self.agent_skill.mu,
            "sigma": self.agent_skill.sigma,
            "trueskill": self.agent_skill.mu - 3 * self.agent_skill.sigma,
        }

import json
import os
from typing import Dict
from typing import Optional

import trueskill

from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir


class LeagueSkillTracker:
    def __init__(
        self,
        tag: str,
        logdir: str,
        default_mu: Optional[float],
        default_sigma: Optional[float],
        resume: Optional[bool] = False,
    ):
        """Helper class for tracking the trueskill of an agent and the league agents.

        Parameters
        ----------
        tag: str
            The name of the training agent.
        logdir: str
            The path to all of the agents.
        default_mu: Optional[float]
            The starting mu for trueskill.
        default_sigma: Optional[float]
            The starting sigma for trueskill.
        resume: Optional[bool]
            Whether or not to load a previous trueskill rating for the training agent.
        """
        self.tag = tag
        self.logdir = logdir
        self.default_mu = 25 if default_mu is None else default_mu
        self.default_sigma = 8.333 if default_sigma is None else default_sigma

        agent_mu = None
        agent_sigma = None
        if resume:
            try:
                epoch = get_most_recent_epoch(os.path.join(logdir, "challengers", tag))
                skill_file = os.path.join(
                    get_save_dir(os.path.join(logdir, "challengers"), tag, epoch),
                    "trueskill.json",
                )
                with open(skill_file, "r") as fp:
                    temp = json.load(fp)
                    agent_mu = temp["mu"]
                    agent_sigma = temp["sigma"]
            except ValueError:
                pass

        self.agent_skill = trueskill.Rating(
            mu=self.default_mu if agent_mu is None else agent_mu,
            sigma=self.default_sigma if agent_sigma is None else agent_sigma,
        )

        self.skill_ratings = self.load_league_skill()

    @classmethod
    def from_args(
        cls,
        args: DotDict,
    ) -> "LeagueSkillTracker":
        """Builds the class from arguments

        Parameters
        ----------
        args: DotDict
            The arguments that can be used to build the class.

        Returns
        -------
        "SkillTracker"
            The class, constructed around the given arguments.
        """
        return cls(
            tag=args.tag,
            logdir=args.logdir,
            default_mu=args.default_mu or 25,
            default_sigma=args.default_sigma or 8.333,
            resume=args.resume or False,
        )

    def load_league_skill(self) -> Dict[str, trueskill.Rating]:
        """Loads the skill ratings of all of the agents from memory.

        Returns
        -------
        Dict[str, trueskill.Rating]
            Stores the trueskill ratings for each agent, with the agent names for keys.
        """
        return {
            agent: self._load_agent_skill(agent)
            for agent in os.listdir(os.path.join(self.logdir, "league"))
        }

    def _load_agent_skill(self, tag: str) -> trueskill.Rating:
        """Checks for the trueskill.json file for the agent specified by tag. If it exists, this
        function loads the file and returns the trueskill. Otherwise, returns the default trueskill.

        Parameters
        ----------
        tag: str
            The name of the agent whose trueskill we want to load.

        Returns
        -------
        trueskill.Rating
            The trueskill rating of the agent specified by tag.
        """
        trueskill_file = os.path.join(self.logdir, "league", tag, "trueskill.json")
        if os.path.exists(trueskill_file):
            with open(trueskill_file, "r") as fp:
                agent_skill = json.load(fp)
        else:
            agent_skill = {"mu": self.default_mu, "sigma": self.default_sigma}
            with open(trueskill_file, "w") as fp:
                json.dump(agent_skill, fp, indent=4)

        return trueskill.Rating(**agent_skill)

    def _save_league_skill(self):
        """Saves the trueskills of all the league agents."""
        league_dir = os.path.join(self.logdir, "league")
        for agent in self.skill_ratings:
            agent_skill = {
                "mu": self.skill_ratings[agent].mu,
                "sigma": self.skill_ratings[agent].sigma,
            }
            with open(os.path.join(league_dir, agent, "trueskill.json"), "w") as fp:
                json.dump(agent_skill, fp, indent=4)

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
        agent_dir = os.path.join(
            self.logdir,
            "challengers",
            self.tag,
            f"{self.tag}_{epoch:05d}",
        )

        if not os.path.isdir(agent_dir):
            os.mkdir(agent_dir)

        trueskill_file = os.path.join(agent_dir, "trueskill.json")
        agent_skill = {
            "mu": self.agent_skill.mu,
            "sigma": self.agent_skill.sigma,
        }
        with open(trueskill_file, "w") as fp:
            json.dump(agent_skill, fp, indent=4)

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
        try:
            _ = self.skill_ratings[opponent]
        except KeyError:
            self.skill_ratings[opponent] = self._load_agent_skill(opponent)
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

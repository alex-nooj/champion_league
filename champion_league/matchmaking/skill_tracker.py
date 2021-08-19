import json
import os
from abc import ABCMeta
from typing import Dict
from typing import Optional

import trueskill
from adept.utils.util import DotDict


class SkillTracker:
    def __init__(
        self,
        tag: str,
        logdir: str,
        default_mu: Optional[int],
        default_sigma: Optional[float],
        resume: Optional[bool] = False,
    ):
        self.tag = tag
        self.logdir = logdir
        self.default_mu = 25 if default_mu is None else default_mu
        self.default_sigma = 8.333 if default_sigma is None else default_sigma

        agent_mu = None
        agent_sigma = None
        if resume:
            agent_path = os.path.join(logdir, "challengers", tag)
            epochs = [
                e.rsplit("_")[-1]
                for e in os.listdir(agent_path)
                if os.path.isdir(os.path.join(agent_path, e)) and e != "sl"
            ]
            if len(epochs) != 0:
                agent_path = os.path.join(
                    logdir,
                    "challengers",
                    tag,
                    f"{tag}_{max(epochs):05d}",
                    "trueskill.json",
                )
                with open(agent_path, "r") as fp:
                    temp = json.load(fp)
                    agent_mu = temp["mu"]
                    agent_sigma = temp["sigma"]

        self.agent_skill = trueskill.Rating(
            mu=self.default_mu if agent_mu is None else agent_mu,
            sigma=self.default_sigma if agent_sigma is None else agent_sigma,
        )

        self.skill_ratings = self.load_league_skill()

    @classmethod
    def from_args(
        cls: ABCMeta,
        args: DotDict,
    ):
        return cls(
            tag=args.tag,
            logdir=args.logdir,
            default_mu=args.default_mu or 25,
            default_sigma=args.default_sigma or 8.333,
            resume=args.resume or False,
        )

    def load_league_skill(self) -> Dict[str, trueskill.Rating]:
        league_dir = os.path.join(self.logdir, "league")
        league_agents = os.listdir(league_dir)
        skill_dict = {}
        for agent in league_agents:
            trueskill_file = os.path.join(
                self.logdir, "league", agent, "trueskill.json"
            )
            if os.path.exists(trueskill_file):
                with open(trueskill_file, "r") as fp:
                    agent_skill = json.load(fp)
            else:
                agent_skill = {"mu": self.default_mu, "sigma": self.default_sigma}
                with open(trueskill_file, "w") as fp:
                    json.dump(agent_skill, fp, indent=4)

            skill_dict[agent] = trueskill.Rating(
                mu=agent_skill["mu"], sigma=agent_skill["sigma"]
            )

        return skill_dict

    def _save_league_skill(self):
        league_dir = os.path.join(self.logdir, "league")
        for agent in self.skill_ratings:
            agent_skill = {
                "mu": self.skill_ratings[agent].mu,
                "sigma": self.skill_ratings[agent].sigma,
            }
            with open(os.path.join(league_dir, agent, "trueskill.json"), "w") as fp:
                json.dump(agent_skill, fp, indent=4)

    def _save_agent_skill(self, epoch: int):
        agent_dir = os.path.join(
            self.logdir,
            "challengers",
            self.tag,
            f"{self.tag}_{epoch:05d}",
            "trueskill.json",
        )
        agent_skill = {
            "mu": self.agent_skill.mu,
            "sigma": self.agent_skill.sigma,
        }
        with open(agent_dir, "w") as fp:
            json.dump(agent_skill, fp, indent=4)

    def save_skill_ratings(self, epoch: int):
        self._save_league_skill()
        self._save_agent_skill(epoch)

    def update(self, agent_won: bool, opponent: str):
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
        return {"mu": self.agent_skill.mu, "sigma": self.agent_skill.sigma}

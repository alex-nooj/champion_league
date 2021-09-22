import json
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import trueskill


class MultiSkillTracker:
    def __init__(
        self,
        agent_paths: List[str],
        mu: Optional[float] = 25,
        sigma: Optional[float] = 8.333,
    ):
        self.default_mu = mu
        self.default_sigma = sigma
        self.agent_skills = self._load_skills(agent_paths)

    def _load_skills(
        self, agent_paths: List[str]
    ) -> Dict[str, Dict[str, Union[trueskill.Rating, str]]]:
        agent_skills = {}
        for path in agent_paths:
            trueskill_file = os.path.join(path, "trueskill.json")
            try:
                with open(trueskill_file, "r") as fp:
                    agent_skill = json.load(fp)
            except FileNotFoundError:
                agent_skill = {"mu": self.default_mu, "sigma": self.default_sigma}
                with open(trueskill_file, "w") as fp:
                    json.dump(agent_skill, fp, indent=4)
            agent_skills[path.rsplit("/")[-1]] = {
                "trueskill": trueskill.Rating(**agent_skill),
                "path": path,
            }
        return agent_skills

    def agent_trueskill(self, tag: str) -> Dict[str, float]:
        skill = self.agent_skills[tag]["trueskill"]
        return {
            "mu": skill.mu,
            "sigma": skill.sigma,
            "trueskill": skill.mu - 3 * skill.sigma,
        }

    def update(self, winner: str, loser: str):
        (
            self.agent_skills[winner]["trueskill"],
            self.agent_skills[loser]["trueskill"],
        ) = trueskill.rate_1vs1(
            self.agent_skills[winner]["trueskill"],
            self.agent_skills[loser]["trueskill"],
        )

    def save_trueskills(self):
        for _, agent in self.agent_skills.items():
            with open(os.path.join(agent["path"], "trueskill.json"), "w") as fp:
                json.dump(
                    {"mu": agent["trueskill"].mu, "sigma": agent["trueskill"].sigma},
                    fp,
                    indent=4,
                )

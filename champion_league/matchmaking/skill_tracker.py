from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import trueskill
from omegaconf import OmegaConf


class SkillTracker:
    def __init__(
        self,
        agents_dir: Path,
        mu: Optional[float] = 25,
        sigma: Optional[float] = 8.333,
    ):
        self.default_mu = mu
        self.default_sigma = sigma
        self.agent_skills = self._load_skills(agents_dir)

    def _load_skills(
        self, agents_dir: Path
    ) -> Dict[str, Dict[str, Union[trueskill.Rating, str]]]:
        agent_skills = {}
        for path in agents_dir.iterdir():
            trueskill_file = path / "trueskill.json"
            try:
                agent_skill = OmegaConf.to_container(OmegaConf.load(trueskill_file))
            except FileNotFoundError:
                agent_skill = {"mu": self.default_mu, "sigma": self.default_sigma}
                OmegaConf.save(config=agent_skill, f=trueskill_file)
            agent_skills[path.stem] = {
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

    def save_skill_ratings(self):
        for _, agent in self.agent_skills.items():
            OmegaConf.save(
                config={"mu": agent["trueksill"].mu, "sigma": agent["trueskill"].sigma},
                f=agent["path"] / "trueskill.yaml",
            )

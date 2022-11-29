from champion_league.agent.scripted.max_base_power import MaxBasePower
from champion_league.agent.scripted.random_actor import RandomActor
from champion_league.agent.scripted.simple_heuristic import SimpleHeuristic


SCRIPTED_AGENTS = {
    "max_base_power_0": MaxBasePower("MaxBasePower"),
    "simple_heuristic_0": SimpleHeuristic("SimpleHeuristic"),
    "random_0": RandomActor("RandomActor"),
}

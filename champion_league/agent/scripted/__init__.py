from champion_league.agent.scripted.max_base_power import MaxBasePower
from champion_league.agent.scripted.random_actor import RandomActor
from champion_league.agent.scripted.simple_heuristic import SimpleHeuristic


SCRIPTED_AGENTS = {
    "MaxBasePower": MaxBasePower("MaxBasePower"),
    "SimpleHeuristic": SimpleHeuristic("SimpleHeuristic"),
    "RandomActor": RandomActor("RandomActor"),
}

from champion_league.reward.rules.binary_faints import AlliedBinaryFaints
from champion_league.reward.rules.binary_faints import OpponentBinaryFaints
from champion_league.reward.rules.hp_diffs import AlliedHPDiff
from champion_league.reward.rules.hp_diffs import OpponentHPDiff
from champion_league.reward.rules.scalar_faints import AlliedScalarFaints
from champion_league.reward.rules.scalar_faints import OpponentScalarFaints
from champion_league.reward.rules.status_rules import AlliedStatusRule
from champion_league.reward.rules.status_rules import OpponentStatusRule
from champion_league.reward.rules.victory_rules import LossRule
from champion_league.reward.rules.victory_rules import VictoryRule

RULES = {
    "OpponentBinaryFaints": OpponentBinaryFaints,
    "AlliedBinaryFaints": AlliedBinaryFaints,
    "OpponentScalarFaints": OpponentScalarFaints,
    "AlliedScalarFaints": AlliedScalarFaints,
    "OpponentStatusRule": OpponentStatusRule,
    "AlliedStatusRule": AlliedStatusRule,
    "OpponentHPDiff": OpponentHPDiff,
    "AlliedHPDiff": AlliedHPDiff,
    "VictoryRule": VictoryRule,
    "LossRule": LossRule,
}

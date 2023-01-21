from enum import auto
from enum import IntEnum


class MoveIdx(IntEnum):
    dmg_1 = 0
    dmg_2 = auto()
    dmg_3 = auto()
    dmg_4 = auto()
    dmg_5 = auto()
    dmg_6 = auto()
    crit_chance = auto()  # Max 6
    acc = auto()
    drain = auto()
    heal = auto()
    pp_ratio = auto()
    recoil = auto()
    # brn = auto()
    # frz = auto()
    # par = auto()
    # psn = auto()
    # slp = auto()
    # tox = auto()
    # con = auto()
    # usr_att = auto()
    # usr_def = auto()
    # usr_spa = auto()
    # usr_spd = auto()
    # usr_spe = auto()
    # usr_acc = auto()
    # usr_switch = auto()
    # tgt_att = auto()
    # tgt_def = auto()
    # tgt_spa = auto()
    # tgt_spd = auto()
    # tgt_spe = auto()
    # tgt_acc = auto()
    # stat_chance = auto()
    # tgt_switch = auto()
    # tgt_trap = auto()
    # flinch = auto()
    # prevents_sound = auto()
    # priority = auto()
    # breaks_protect = auto()
    # protects = auto()
    # light_screen = auto()
    # reflect = auto()
    # aurora_veil = auto()
    # spikes = auto()
    # tox_spikes = auto()
    # stealth_rock = auto()
    # substitute = auto()
    # taunt = auto()
    # health = auto()
    # stat_reset = auto()
    # encore = auto()
    # clears_tgt_hazards = auto()
    # clears_usr_hazards = auto()
    physical = auto()
    special = auto()
    status = auto()


# MOVE_EFFECTS = {
#     "Acrobatics": {},
#     "Air Slash": {
#         MoveIdx.flinch: 0.3,
#     },
#     "Ancient Power": {
#         MoveIdx.usr_att: 1/6,
#         MoveIdx.usr_def: 1/6,
#         MoveIdx.usr_spa: 1/6,
#         MoveIdx.usr_spd: 1/6,
#         MoveIdx.usr_spe: 1/6,
#         MoveIdx.stat_chance: 0.1,
#     },
#     "Aqua Jet": {
#         MoveIdx.priority: 1 / 5,
#     },
#     "Aurora Veil": {
#         MoveIdx.aurora_veil: 1.0,
#     },
#     "Beat Up": {},
#     "Belly Drum": {MoveIdx.health: 0.5, MoveIdx.usr_att: 1.0, MoveIdx.stat_chance: 1.0},
#     "Blizzard": {
#         MoveIdx.frz: 0.1,
#     },
#     "Body Press": {},
#     "Bolt Beak": {},
#     "Boomburst": {},
#     "Bug Buzz": {MoveIdx.tgt_spd: -1 / 6, MoveIdx.stat_chance: 0.1},
#     "Bulk Up": {MoveIdx.usr_att: 1 / 6, MoveIdx.usr_def: 1 / 6, MoveIdx.stat_chance: 1.0},
#     "Bullet Punch": {
#         MoveIdx.priority: 1 / 5,
#     },
#     "Calm Mind": {MoveIdx.usr_spa: 1 / 6, MoveIdx.usr_spd: 1 / 6, MoveIdx.stat_chance: 1.0},
#     "Clear Smog": {MoveIdx.stat_reset: 1.0, MoveIdx.stat_chance: 1.0},
#     "Close Combat": {MoveIdx.usr_def: -1 / 6, MoveIdx.usr_spd: -1 / 6, MoveIdx.stat_chance: 1.0},
#     "Crabhammer": {},
#     "Crunch": {
#         MoveIdx.tgt_def: -1 / 6,
#         MoveIdx.stat_chance: 0.2,
#     },
#     "Dark Pulse": {
#         MoveIdx.flinch: 0.2,
#     },
#     "Defog": {
#         MoveIdx.clears_tgt_hazards: 1.0,
#         MoveIdx.clears_usr_hazards: 1.0,
#     },
#     "Discharge": {
#         MoveIdx.par: 0.3,
#     },
#     "Draco Meteor": {
#         MoveIdx.usr_spa: -2 / 6,
#         MoveIdx.stat_chance: 1.0,
#     },
#     "Dragon Dance": {
#         MoveIdx.usr_att: 1 / 6,
#         MoveIdx.usr_spe: 1 / 6,
#         MoveIdx.stat_chance: 1.0,
#     },
#     "Draining Kiss": {},
#     "Dual Wingbeat": {},
#     "Earth Power": {
#         MoveIdx.tgt_spd: -1 / 6,
#         MoveIdx.stat_chance: 0.1,
#     },
#     "Earthquake": {},
#     "Encore": {
#         MoveIdx.encore: 1.0,
#     },
#     "Facade": {},
#     "Final Gambit": {
#         MoveIdx.health: 1.0,
#     },
#     "Fire Blast": {
#         MoveIdx.brn: 0.1,
#     },
#     "Fire Fang": {
#         MoveIdx.brn: 0.1,
#         MoveIdx.flinch: 0.1,
#     },
#     "Fire Punch": {
#         MoveIdx.brn: 0.1,
#     },
#     "Fire Spin": {},
#     "Flamethrower": {
#         MoveIdx.brn: 0.1,
#     },
#     "Flare Blitz": {
#         MoveIdx.brn: 0.1,
#     },
#     "Flash Cannon": {
#         MoveIdx.tgt_spd: -1 / 6,
#         MoveIdx.stat_chance: 0.1,
#     },
#     "Flip Turn": {MoveIdx.usr_switch: 1.0},
#     "Focus Blast": {
#         MoveIdx.tgt_spd: -1 / 6,
#         MoveIdx.stat_chance: 0.1,
#     },
#     "Foul Play": {},
#     "Freeze-Dry": {
#         MoveIdx.frz: 0.1,
#     },
#     "Future Sight": {},
#     "Giga Drain": {},
#     "Grassy Glide": {},
#     "Growth": {MoveIdx.usr_att: 1 / 6, MoveIdx.usr_spa: 1 / 6, MoveIdx.stat_chance: 1.0},
#     "Hail": {},
#     "Haze": {
#         MoveIdx.stat_reset: 1.0,
#         MoveIdx.stat_chance: 1.0,
#     },
#     "Heal Bell": {},
#     "Heat Wave": {
#         MoveIdx.brn: 0.1,
#     },
#     "Hex": {},
#     "Hydro Pump": {},
#     "Hypnosis": {MoveIdx.slp: 1.0},
#     "Ice Beam": {
#         MoveIdx.frz: 0.1,
#     },
#     "Ice Punch": {
#         MoveIdx.frz: 0.1,
#     },
#     "Ice Shard": {
#         MoveIdx.priority: 1 / 5,
#     },
#     "Icicle Crash": {MoveIdx.flinch: 0.3},
#     "Icicle Spear": {},
#     "Iron Defense": {MoveIdx.usr_def: 2 / 6, MoveIdx.stat_chance: 1.0},
#     "Iron Head": {MoveIdx.flinch: 0.3},
#     "Knock Off": {},
#     "Lava Plume": {
#         MoveIdx.brn: 0.3,
#     },
#     "Leech Seed": {},
#     "Light Screen": {MoveIdx.light_screen: 1.0},
#     "Liquidation": {MoveIdx.tgt_def: -1 / 6, MoveIdx.stat_chance: 0.2},
#     "Low Kick": {},
#     "Mach Punch": {
#         MoveIdx.priority: 1 / 5,
#     },
#     "Meteor Beam": {MoveIdx.usr_spa: 1/6, MoveIdx.stat_chance: 1.0},
#     "Moonblast": {MoveIdx.tgt_spa: -1 / 6, MoveIdx.stat_chance: 0.3},
#     "Mystical Fire": {MoveIdx.tgt_spa: -1 / 6, MoveIdx.stat_chance: 1.0},
#     "Nasty Plot": {MoveIdx.usr_spa: 2 / 6, MoveIdx.stat_chance: 1.0},
#     "Overdrive": {},
#     "Overheat": {MoveIdx.usr_spa: -2 / 6, MoveIdx.stat_chance: 1.0},
#     "Pain Split": {},
#     "Plasma Fists": {},
#     "Play Rough": {MoveIdx.tgt_att: -1 / 6, MoveIdx.stat_chance: 0.1},
#     "Poison Jab": {MoveIdx.psn: 0.3},
#     "Poltergeist": {},
#     "Power Whip": {},
#     "Protect": {MoveIdx.protects: 1.0},
#     "Psychic": {MoveIdx.tgt_spd: -1 / 6, MoveIdx.stat_chance: 0.1},
#     "Psychic Fangs": {MoveIdx.clears_tgt_hazards: 1.0},
#     "Psyshock": {},
#     "Quiver Dance": {
#         MoveIdx.usr_spa: 1 / 6,
#         MoveIdx.usr_spd: 1 / 6,
#         MoveIdx.usr_spe: 1 / 6,
#         MoveIdx.stat_chance: 1.0,
#     },
#     "Rapid Spin": {MoveIdx.clears_usr_hazards: 1.0},
#     "Recover": {MoveIdx.health: -0.5},
#     "Reflect": {MoveIdx.reflect: 1.0},
#     "Rock Blast": {},
#     "Roost": {MoveIdx.health: -0.5},
#     "Sand Tomb": {MoveIdx.tgt_trap: 1.0},
#     "Scald": {MoveIdx.brn: 0.3},
#     "Scale Shot": {MoveIdx.usr_spd: 1 / 6, MoveIdx.usr_def: -1 / 6, MoveIdx.stat_chance: 0.15},
#     "Seismic Toss": {},
#     "Shadow Ball": {MoveIdx.tgt_spd: -1 / 6, MoveIdx.stat_chance: 0.2},
#     "Shadow Sneak": {
#         MoveIdx.priority: 1 / 5,
#     },
#     "Slack Off": {MoveIdx.health: -0.5},
#     "Sleep Powder": {MoveIdx.slp: 1.0},
#     "Sludge Bomb": {MoveIdx.psn: 0.3},
#     "Sludge Wave": {MoveIdx.psn: 0.1},
#     "Soft-Boiled": {MoveIdx.health: -0.5},
#     "Spikes": {MoveIdx.spikes: 1.0},
#     "Spirit Break": {MoveIdx.tgt_spa: -1 / 6, MoveIdx.stat_chance: 1.0},
#     "Spore": {MoveIdx.slp: 1.0},
#     "Stealth Rock": {MoveIdx.stealth_rock: 1.0},
#     "Sticky Web": {},
#     "Stone Edge": {},
#     "Substitute": {MoveIdx.substitute: 1.0, MoveIdx.health: 0.25},
#     "Sucker Punch": {},
#     "Superpower": {MoveIdx.usr_att: -1 / 6, MoveIdx.usr_def: -1 / 6, MoveIdx.stat_chance: 1.0},
#     "Surf": {},
#     "Surging Strikes": {},
#     "Swords Dance": {MoveIdx.usr_att: 2 / 6, MoveIdx.stat_chance: 1.0},
#     "Taunt": {MoveIdx.taunt: 1.0},
#     "Teleport": {MoveIdx.usr_switch: 1.0},
#     "Thunder": {MoveIdx.par: 0.3},
#     "Thunder Fang": {
#         MoveIdx.par: 0.1,
#         MoveIdx.flinch: 0.1,
#     },
#     "Thunder Punch": {MoveIdx.par: 0.1},
#     "Thunder Wave": {MoveIdx.par: 1.0},
#     "Thunderbolt": {MoveIdx.par: 0.1},
#     "Toxic": {MoveIdx.tox: 1.0},
#     "Toxic Spikes": {MoveIdx.tox_spikes: 1.0},
#     "Transform": {},
#     "Triple Axel": {},
#     "U-turn": {MoveIdx.usr_switch: 1.0},
#     "Volt Switch": {MoveIdx.usr_switch: 1.0},
#     "Weather Ball": {},
#     "Will-O-Wisp": {MoveIdx.brn: 1.0},
#     "Wood Hammer": {},
# }
#
# MOVE_EFFECTS = {k.lower().replace(" ", "").replace("-", ""): v for k, v in MOVE_EFFECTS.items()}

from champion_league.preprocessor import Preprocessor


class EmbeddingPreprocessor(Preprocessor):
    def __init__(self):
        active_keys = [
            "species",
            "ability",
            "item",
            "burn",
            "faint",
            "freeze",
            "paralyze",
            "poisoned",
            "sleep",
            "toxic",
            "hp_fraction",
            "atk_stat",
            "def_stat",
            "spa_stat",
            "spd_stat",
            "spe_stat",
            "acc_boost",
            "eva_boost",
        ]

        move_keys = [
            "move_name",
            "dmg",
        ]

        weather = ["hail", "raindance", "sandstorm", "sunnyday"]
        conditions = [
            "aurora_veil",
            "light_screen",
            "lucky_chant",
            "mist",
            "reflect",
            "safeguard",
            "spikes",
            "stealth_rock",
            "sticky_web",
            "tailwind",
            "toxic_spikes",
        ]
        field = [
            "electric_terrain",
            "grassy_terrain",
            "gravity",
            "heal_block",
            "magic_room",
            "misty_terrain",
            "psychic_terrain",
            "trick_room",
            "wonder_room",
        ]

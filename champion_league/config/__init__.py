# MIT License. 2020. Joe Tatusko.
import pathlib
import sys
from os import path

from . import cfg_helper

# Global paths
MODULE_PATH = pathlib.Path(__file__).parent.parent.absolute()
PROJECT_PATH = MODULE_PATH.parent.absolute()


def __get_base_cfg_path() -> str:
    return path.join(MODULE_PATH, "config", "base.yml")


def __get_script_cfg_path() -> str:
    script_path, _ = path.splitext(sys.argv[0])
    script_name = path.split(script_path)[-1]
    return path.join(MODULE_PATH, "config", script_name + ".yml")


# Global config object
CFG = cfg_helper.load_cfg(__get_base_cfg_path(), __get_script_cfg_path())

__all__ = ["CFG"]

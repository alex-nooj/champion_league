from champion_league.preprocessors.base_preprocessor import Preprocessor
from champion_league.preprocessors.frame_stacker import FrameStacker
from champion_league.preprocessors.modular_preprocessor import ModularPreprocessor
from champion_league.preprocessors.simple_preprocessor import SimplePreprocessor
from champion_league.utils.directory_utils import DotDict

preprocessors = {
    "ModularPreprocessor": ModularPreprocessor,
    "SimplePreprocessor": SimplePreprocessor,
    "FrameStacker": FrameStacker,
}


def build_preprocessor_from_args(args: DotDict) -> Preprocessor:
    return preprocessors[args.preprocessor].from_args(args)

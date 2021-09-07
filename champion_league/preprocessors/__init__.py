from champion_league.utils.directory_utils import DotDict
from champion_league.preprocessors.base_preprocessor import Preprocessor
from champion_league.preprocessors.modular_preprocessor import ModularPreprocessor
from champion_league.preprocessors.simple_preprocessor import SimplePreprocessor

preprocessors = {
    "ModularPreprocessor": ModularPreprocessor,
    "SimplePreprocessor": SimplePreprocessor,
}


def build_preprocessor_from_args(args: DotDict) -> Preprocessor:
    return preprocessors[args.preprocessor].from_args(args)

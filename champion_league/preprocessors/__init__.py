from adept.utils.util import DotDict

from champion_league.preprocessors.base_preprocessor import Preprocessor


def build_preprocessor_from_args(args: DotDict) -> Preprocessor:
    import importlib

    preprocessor_path = f"champion_league.preprocessors.{args.preprocessor}"
    module = importlib.import_module(preprocessor_path)
    return module.build_from_args(args)

from champion_league.preprocessors.base_preprocessor import Preprocessor
from champion_league.preprocessors.modular_preprocessor import ModularPreprocessor
from champion_league.preprocessors.simple_preprocessor import SimplePreprocessor

# from champion_league.preprocessors.frame_stacker import FrameStacker

PREPROCESSORS = {
    ModularPreprocessor.__name__: ModularPreprocessor,
    SimplePreprocessor.__name__: SimplePreprocessor,
    # FrameStacker.__name__: FrameStacker,
}

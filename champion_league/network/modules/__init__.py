from champion_league.network.modules.conv import Conv
from champion_league.network.modules.embedding import Embedding
from champion_league.network.modules.encoder import Encoder
from champion_league.network.modules.lstm import LSTMMod

MODULES = {
    Conv.__name__: Conv,
    Embedding.__name__: Embedding,
    Encoder.__name__: Encoder,
    LSTMMod.__name__: LSTMMod,
}

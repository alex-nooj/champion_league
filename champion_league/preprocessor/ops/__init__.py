from champion_league.preprocessor.ops.add_batch_dim import AddBatchDim
from champion_league.preprocessor.ops.embed_active import EmbedActive
from champion_league.preprocessor.ops.embed_moves import EmbedMoves
from champion_league.preprocessor.ops.embed_team import EmbedTeam

OPS = {
    EmbedActive.__name__: EmbedActive,
    EmbedMoves.__name__: EmbedMoves,
    EmbedTeam.__name__: EmbedTeam,
    AddBatchDim.__name__: AddBatchDim,
}

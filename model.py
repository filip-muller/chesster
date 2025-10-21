import chess
from torch import nn


class PositionEvaluator(nn.Module):
    def __init__(self, input_size=132):
        super().__init__()
        layer_sizes = [256, 512, 128]
        previous_size = input_size
        self.fc_layers_list = []
        for layer_size in layer_sizes:
            self.fc_layers_list.append(nn.Linear(previous_size, layer_size))
            self.fc_layers_list.append(nn.GELU())
            previous_size = layer_size

        last_layer_size = layer_sizes[-1]

        self.fc_layers = nn.Sequential(*self.fc_layers_list)

        self.classification_layer = nn.Linear(last_layer_size, 1)

    def forward(self, x):
        res = self.fc_layers(x)
        res = self.classification_layer(res)
        return res


def evaluate_position(fen):
    return evaluate_position_piece_value(fen)


def min_max_eval(fen, depth=None):
    pass

def evaluate_position_piece_value(fen):
    """
    Evaluates position simply using standard piece values

    Doesnt even take potential recaptures into consideration, this is to be done by min_max

    Result is (white_material - black_material) -> positive values white lead, negative black
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    board = chess.Board(fen)
    white_material = 0
    black_material = 0

    for piece_type in piece_values:
        white_material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        black_material += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    return white_material - black_material

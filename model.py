import chess
import torch
from torch import nn
from fen_to_vec import fen_to_vec


class NNModel(nn.Module):
    def __init__(self, input_size=133):
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


class PositionEvaluator:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = "weights/full_pieces.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NNModel().to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def evaluate_position(self, fen):
        return self.evaluate_positions([fen])[0].item()

    def evaluate_positions(self, fens):
        """Uses batching, use for better performance"""
        batch_size = 2048
        res = []
        for beg in range(0, len(fens), batch_size):
            vec = torch.stack([fen_to_vec(fen) for fen in fens[beg:beg + 2048]]).to(self.device)
            with torch.no_grad():
                model_evals = self.model(vec)
            res.append(model_evals.flatten().tolist())
        return res



def min_max_eval(evaluator, fen, depth=None):
    if depth is None:
        depth = 5
    board = chess.Board(fen)
    moves = board.legal_moves

    for move in moves:
        new_board = board.copy()
        new_board.push(move)



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

import chess
import torch
from torch import nn
from fen_to_vec import fen_to_vec


class NNModel(nn.Module):
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


class PositionEvaluator:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = "weights/new/900_plus_1500.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NNModel().to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def evaluate_position(self, fen):
        return self.evaluate_positions([fen])[0]

    def evaluate_positions(self, fens, batch_size=2048):
        """Uses batching, use for better performance"""
        res = []
        for beg in range(0, len(fens), batch_size):
            vec = torch.stack([fen_to_vec(fen) for fen in fens[beg:beg + batch_size]]).to(self.device)
            with torch.no_grad():
                model_evals = self.model(vec)
            res.extend(model_evals.flatten().tolist())
        return res


def min_max_eval(fen, depth=None, always_continue_with_captures=None, evaluator=None):
    """Returns tuple (position_eval, best_move)"""
    if depth is None:
        depth = 5
    if always_continue_with_captures is None:
        always_continue_with_captures = False
    if evaluator is None:
        evaluator = PositionEvaluator()

    board = chess.Board(fen)

    if depth == 0:
        res = evaluator.evaluate_position(fen)
        if board.turn == chess.BLACK:
            # return score from white perspective, model returns from color to move
            res *= -1
        return res

    # white wants to maximize, black minimize (negative evals)
    maximizing = board.turn == chess.WHITE

    return _min_max_rec(evaluator, board, depth, always_continue_with_captures, maximizing)


def _min_max_rec(evaluator: PositionEvaluator, board: chess.Board, depth: int, always_continue_with_captures: bool, maximizing: bool):
    """Returns tuple (position_eval, best_move)"""
    if depth < 1:
        raise ValueError("Depth cannot be lower than 1")
    if board.is_game_over():
        # if game is over return based on the game result
        result = board.result()
        res_to_val = {
            "1-0": 100,
            "0-1": -100,
            "1/2-1/2": 0,
        }
        val = res_to_val[result.strip()]
        return (val, None)
    if board.is_repetition(3):
        print("Returning repetition")
        return (0, None)
    if board.is_fifty_moves():
        return (0, None)
    color_to_move = board.turn

    min_or_max = max if maximizing else min
    possible_boards : list[chess.Board] = []
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        new_board = board.copy()
        new_board.push(move)
        possible_boards.append(new_board)
    if depth > 1:
        # find recursively, get just the position evaluation (index 0)
        evals = [_min_max_rec(evaluator, b, depth - 1, not maximizing)[0] for b in possible_boards]
    else:
        fens = [b.fen() for b in possible_boards]
        evals = evaluator.evaluate_positions(fens)
        if color_to_move == chess.WHITE:
            # evals came from a position where BLACK is to move (it is one move after the one with white to move)
            # thus we need to flip evals from the perspective of color to move to perspective of white
            evals = [-1 * val for val in evals]
        # note: these positions also get evaluated twice (also by the evaluator, improve later for performance)
        for i in range(len(possible_boards)):
            b = possible_boards[i]
            for legal_move in b.legal_moves:
                # if we find a capture of a non-pawn piece, keep evaluating deeper (ignoring depth limits)
                if b.is_capture(legal_move):
                    piece_to_capture = b.piece_at(move.to_square)
                    # piece_at can probably return None for en passant
                    if piece_to_capture is not None and piece_to_capture.piece_type != chess.PAWN:
                        # do not decrease depth, it is already 1, this will keep it going while captures are available
                        val = _min_max_rec(evaluator, b, depth, not maximizing)[0]
                        evals[i] = val

    evals_with_moves = list(zip(evals, legal_moves))
    # return tuple (eval, best_move)
    return min_or_max(evals_with_moves, key=lambda x: x[0])


def find_best_move(fen, depth=None, always_continue_with_captures=None):
    board = chess.Board(fen)
    if board.is_game_over():
        return None

    evaluation, best_move = min_max_eval(fen, depth, always_continue_with_captures)
    print(f"Selecting move {best_move} with eval {evaluation} (depth {depth})")
    return best_move


def evaluate_position_piece_value(fen, to_move_perspective=False):
    """
    Evaluates position simply using standard piece values

    Doesnt even take potential recaptures into consideration, this is to be done by min_max

    Result is (white_material - black_material) -> positive values white lead, negative black

    If to_move_perspective is True, instead returns the value from the perspective of the player to move.
    So if it is black to move, it will instead return (black_material - white_material)
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

    res = white_material - black_material

    if to_move_perspective and board.turn == chess.BLACK:
        # if we want value from position of player to move and its black turn, flip value
        res *= -1
    return res

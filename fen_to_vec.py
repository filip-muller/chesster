"""Converts a FEN of a position into a vector that serves as the input of the model"""
import torch

# TODO: include en passant, potentially by a new piece "pawn capturable by en passant"
# TODO: halfmoves since last capture or pawnmove not included (relevant for draw by 50-move rule)

EMPTY = 0  # square with no piece
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6  # maybe king should have a lower value and queen should be most valuable

# enums for onehot
PAWN_IDX = 0
KNIGHT_IDX = 1
BISHOP_IDX = 2
ROOK_IDX = 3
QUEEN_IDX = 4
KING_IDX = 5

COLOR_TO_MOVE_IDX = 6
COLOR_NOT_TO_MOVE_IDX = 7

# EMPTY_IDX = 8  # square with no piece - often not included

ONEHOT_ENCODING_SIZE = 8  # should be 9 if EMPTY_IDX is used - hardcoded for now


WHITE = 1
BLACK = -1

# alligns with legacy as COLOR_TO_MOVE == WHITE and NOT_TO_MOVE = BLACK
COLOR_TO_MOVE = 1
COLOR_NOT_TO_MOVE = -1
NEUTRAL = 0

CANNOT_CASTLE = 0
CAN_CASTLE = 1

FEN_CODES = {
    "P": PAWN,
    "N": KNIGHT,
    "B": BISHOP,
    "R": ROOK,
    "Q": QUEEN,
    "K": KING,
}

FEN_CODES_ONEHOT = {
    "P": PAWN_IDX,
    "N": KNIGHT_IDX,
    "B": BISHOP_IDX,
    "R": ROOK_IDX,
    "Q": QUEEN_IDX,
    "K": KING_IDX,
}

def create_piece_conversion_dict(onehot=False):
    conversion = {}
    fen_codes = FEN_CODES if not onehot else FEN_CODES_ONEHOT
    for char, val in fen_codes.items():
        conversion[char] = [val, "white"]
        # lowercase used for black characters in FEN
        conversion[char.lower()] = [val, "black"]
    empty_representation = [EMPTY, NEUTRAL] if not onehot else []
    conversion["_"] = empty_representation.copy()
    return conversion


PIECE_CONVERSION_DICT = create_piece_conversion_dict()
PIECE_CONVERSION_DICT_ONEHOT = create_piece_conversion_dict(onehot=True)


def from_color_to_to_move(color, color_to_move, onehot=False):
    """Values which are not "black" or "white" are left as they are"""
    if color_to_move not in ("black", "white"):
        raise ValueError("Wrong color")

    color_to_move_const = COLOR_TO_MOVE if not onehot else COLOR_TO_MOVE_IDX
    color_not_to_move_const = COLOR_NOT_TO_MOVE if not onehot else COLOR_NOT_TO_MOVE_IDX

    color_not_to_move = "black" if color_to_move == "white" else "white"
    if color == color_to_move:
        return color_to_move_const
    if color == color_not_to_move:
        return color_not_to_move_const
    return color


def castling_fen_to_vec(castling_fen, color_to_move):
    """Takes the part of fen that talks about castling (e.g. KQkq) and converts to list"""
    if color_to_move not in ("white", "black"):
        raise ValueError(f"Invalid color '{color_to_move}'")
    res = []
    # first look at color to move castling
    char_sequence = "KQkq" if color_to_move == "white" else "kqKQ"
    for char in char_sequence:
        if char in castling_fen:
            res.append(CAN_CASTLE)
        else:
            res.append(CANNOT_CASTLE)
    return res


def fen_to_vec(fen):
    """
    Currently a piece is represented by a single number, as PIECE + COLOR.
    For example, black bishop is 3 + 100 = 103
    Output:

    items 0-127 - pieces by row, first item pieces type, second color (to move or not to move, neutral for empty)
    128 - color to move kingside castle
    129 - color to move queenside castle
    130 - color not to move kingside castle
    131 - color not to move queenside castle
    """
    color_to_move_fen = fen.split(" ")[1].strip()
    if color_to_move_fen == "w":
        color_to_move = "white"
    elif color_to_move_fen == "b":
        color_to_move = "black"
    else:
        raise ValueError(f"Invalid color to move in fen: '{color_to_move_fen}'")

    pieces_fen = fen.split(" ")[0]
    for digit in range(1, 9):
        # expand empty squares and fill with _ characters
        pieces_fen = pieces_fen.replace(str(digit), "_" * digit)
    # delete row seperators
    pieces_fen = pieces_fen.replace("/", "")

    pieces = [PIECE_CONVERSION_DICT[ch] for ch in pieces_fen]
    pieces_by_to_move = [(piece_type, from_color_to_to_move(col, color_to_move)) for piece_type, col in pieces]
    pieces_vec = [e for piece in pieces_by_to_move for e in piece]

    castling_fen = fen.split(" ")[2]
    castling_vec = castling_fen_to_vec(castling_fen, color_to_move=color_to_move)

    res = pieces_vec + castling_vec

    return torch.tensor(res, dtype=torch.float)


def fen_to_onehot(fen):
    """
    Uses onehot encoding for both piece type and piece color

    Returns list of pieces (ignores empty squares)
    """
    color_to_move_fen = fen.split(" ")[1].strip()
    if color_to_move_fen == "w":
        color_to_move = "white"
    elif color_to_move_fen == "b":
        color_to_move = "black"
    else:
        raise ValueError(f"Invalid color to move in fen: '{color_to_move_fen}'")

    pieces_fen = fen.split(" ")[0]
    for digit in range(1, 9):
        # expand empty squares and fill with _ characters
        pieces_fen = pieces_fen.replace(str(digit), "_" * digit)
    # delete row seperators
    pieces_fen = pieces_fen.replace("/", "")

    # only use pieces, not empty squares
    pieces = [(PIECE_CONVERSION_DICT_ONEHOT[ch], pos) for pos, ch in enumerate(pieces_fen) if PIECE_CONVERSION_DICT_ONEHOT[ch]]
    pieces_by_to_move = [(piece_type, from_color_to_to_move(col, color_to_move, onehot=True), pos) for (piece_type, col), pos in pieces]

    pieces_vec = [[0 if i not in (piece_idx, color_idx) else 1 for i in range(ONEHOT_ENCODING_SIZE)] + [pos] for piece_idx, color_idx, pos in pieces_by_to_move]

    return torch.tensor(pieces_vec, dtype=torch.float)


def batch_and_pad_onehot(onehot_encoded_boards: list[torch.Tensor]):
    """Input: List of 2D tensors each of shape [piece count (different for each tensor), one_hot_encoding_dim]"""
    max_pieces = max(t.shape[0] for t in onehot_encoded_boards)

    padded = [torch.nn.functional.pad(t, (0, 0, 0, max_pieces - t.shape[0])) for t in onehot_encoded_boards]
    batched = torch.stack(padded)

    piece_counts = [t.shape[0] for t in onehot_encoded_boards]

    return batched, piece_counts

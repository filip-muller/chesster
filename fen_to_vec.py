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
    "K": KING
}

def create_piece_conversion_dict():
    conversion = {}
    for char, val in FEN_CODES.items():
        conversion[char] = [val, "white"]
        # lowercase used for black characters in FEN
        conversion[char.lower()] = [val, "black"]
    conversion["_"] = [EMPTY, NEUTRAL]
    return conversion


PIECE_CONVERSION_DICT = create_piece_conversion_dict()


def from_color_to_to_move(color, color_to_move):
    # Values which are not "black" or "white" are left as they are
    if color_to_move not in ("black", "white"):
        raise ValueError("Wrong color")
    color_not_to_move = "black" if color_to_move == "white" else "white"
    if color == color_to_move:
        return COLOR_TO_MOVE
    if color == color_not_to_move:
        return COLOR_NOT_TO_MOVE
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

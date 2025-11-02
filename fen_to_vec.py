"""Converts a FEN of a position into a vector that serves as the input of the model"""
import torch

# TODO: include en passant, potentially by a new piece "pawn capturable by en passant"
# TODO: halfmoves since last capture or pawnmove not included (relevant for draw by 50-move rule)

# TODO probably make embedding such that it doesnt differentiate between white/black pieces but between pieces of the player to move and the opponent
# This removes redundancy - you wont create a new position (in terms of embedding) just by swapping the colors
# This will make it such that model always evaluates from the perspective of the player to move (positive number means player to move is winning, not white), keep that in mind !!

EMPTY = 0  # square with no piece
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6  # maybe king should have a lower value and queen should be most valuable

WHITE = 1
BLACK = -1
NEUTRAL = 0  # for empty squares

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
        conversion[char] = [val, WHITE]
        # lowercase used for black characters in FEN
        conversion[char.lower()] = [val, BLACK]
    conversion["_"] = [EMPTY, NEUTRAL]
    return conversion


PIECE_CONVERSION_DICT = create_piece_conversion_dict()


def castling_fen_to_vec(castling_fen):
    """Takes the part of fen that talks about castling (e.g. KQkq) and converts to list"""
    res = []
    for char in "KQkq":
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

    items 0-127 - pieces by row, first item pieces type, second color, white for empty
    128 - color to move
    129 - white kingside castle
    130 - white queenside castle
    131 - black kingside castle
    132 - white queenside castle
    """
    pieces_fen = fen.split(" ")[0]
    # print(pieces_fen)
    for digit in range(1, 9):
        # expand empty squares and fill with _ characters
        pieces_fen = pieces_fen.replace(str(digit), "_" * digit)
    # delete row seperators
    pieces_fen = pieces_fen.replace("/", "")

    pieces_vec = [e for ch in pieces_fen for e in PIECE_CONVERSION_DICT[ch]]

    color_to_move_fen = fen.split(" ")[1].strip()
    if color_to_move_fen == "w":
        color_to_move_vec = [WHITE]
    elif color_to_move_fen == "b":
        color_to_move_vec = [BLACK]
    else:
        raise ValueError(f"Invalid color to move in fen: '{color_to_move_fen}'")

    castling_fen = fen.split(" ")[2]
    castling_vec = castling_fen_to_vec(castling_fen)

    res = pieces_vec + color_to_move_vec + castling_vec

    return torch.tensor(res, dtype=torch.float)

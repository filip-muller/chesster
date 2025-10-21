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
KING = 6

WHITE = 0
BLACK = 100

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
    conversion["_"] = [EMPTY, WHITE]
    return conversion


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

    items 0-63 - pieces by row, starting in row 8 - h1, g1, ..., h2, ..., a8
    NEW: items 0-127 - pieces by row, first item pieces type, second color, white for empty
    65 - white kingside castle
    66 - white queenside castle
    67 - black kingside castle
    68 - white queenside castle
    """
    pieces_fen = fen.split(" ")[0]
    # print(pieces_fen)
    for digit in range(1, 9):
        # expand empty squares and fill with _ characters
        pieces_fen = pieces_fen.replace(str(digit), "_" * digit)
    # delete row seperators
    pieces_fen = pieces_fen.replace("/", "")

    conversion = create_piece_conversion_dict()

    pieces_vec = []
    for ch in pieces_fen:
        pieces_vec.extend(conversion[ch])

    castling_fen = fen.split(" ")[2]
    castling_vec = castling_fen_to_vec(castling_fen)

    res = pieces_vec + castling_vec

    return torch.tensor(res, dtype=torch.float)

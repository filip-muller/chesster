import itertools

import chess


def create_positions(piece_type=None):
    if piece_type is None:
        piece_type = chess.QUEEN
    res = []
    # white king always in the top left quarter of the chessboard (can always be rotated this way)
    white_king_options = [8*i + j for i in range(4) for j in range(4)]
    black_king_options = list(range(64))
    piece_options = list(range(64))
    turn_options = (chess.WHITE, chess.BLACK)
    for positions in itertools.product(white_king_options, black_king_options, piece_options, turn_options):
        white_king_position, black_king_position, piece_position, turn = positions
        board = chess.Board(None)
        board.turn = turn
        board.set_piece_at(white_king_position, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(black_king_position, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(piece_position, chess.Piece(piece_type, chess.WHITE))
        if board.is_valid():
            res.append(board)

    return res


def mark_won_positions(fens, boards, scores: dict):
    for fen, boards in zip(fens, boards):
        if chess.Board(fen).is_checkmate():
            # mate in zero
            scores[fen] = 0

def mark_previous_positions():
    # TODO
    pass


def create_positions_file(path):
    res = create_positions()
    print(len(res))
    with open(path, "w") as f:
        for b in res:
            # information about castling en passant and move/ply number is irrelevant
            shorter_fen = b.fen().split("-")[0].strip()
            f.write(shorter_fen + "\n")


def read_fens(path):
    with open(path) as f:
        lines = f.readlines()
    res = [fen.strip() for fen in lines if fen.strip()]
    return res


if __name__ == "__main__":
    # create_positions_file("positions.txt")

    fens = read_fens("positions.txt")
    boards = [chess.Board(fen) for fen in fens]
    scores = {}
    mark_won_positions(fens, boards, scores)
    print(len(scores))
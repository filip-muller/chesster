import random
import chess
import torch

from fen_to_vec import fen_to_vec
from model import PositionEvaluator, evaluate_position_piece_value


def create_random_fen():
    """Creates fens of positions that dont necessarily make sense, just for piece value"""
    fen_ending = "- 0 1"
    # give different castling rights with 50 percent chance
    castling = "".join([char for char in "KQkq" if random.random() > 0.5])
    if not castling:
        castling = "-"
    color_turn = "w" if random.random() > 0.5 else "b"
    fen_ending = color_turn + " " + castling + " " + fen_ending

    pieces = "PNBRQ"
    pieces += pieces.lower()
    pieces += "_"
    weights = [1 for _ in pieces]
    # higher weight for empty square
    weights[-1] = 10
    pieces_fen = list(random.choices(pieces, weights=weights, k=64))
    black_king = random.randint(0, 63)
    white_king = random.randint(0, 63)
    if white_king == black_king:
        # make sure we dont go over 63
        black_king = (white_king + 1) % 64
    pieces_fen[white_king] = "K"
    pieces_fen[black_king] = "k"
    pieces_fen = "/".join("".join(pieces_fen[beg:beg + 8]) for beg in range(0, 64, 8))
    for ln in range(8, 0, -1):
        pieces_fen = pieces_fen.replace("_" * ln, str(ln))

    res = pieces_fen + " " + fen_ending
    return res


def create_simple_random_fen(piece_count=1):
    fen_ending = "- 0 1"
    # give different castling rights with 50 percent chance
    castling = "".join([char for char in "KQkq" if random.random() > 0.5])
    if not castling:
        castling = "-"
    color_turn = "w" if random.random() > 0.5 else "b"
    fen_ending = color_turn + " " + castling + " " + fen_ending

    pieces_fen = ["_" for _ in range(64)]
    if piece_count is None:
        piece_count = random.randint(0, 40)
    for _ in range(piece_count):
        pawn_index = random.randint(0, 63)
        pieces_fen[pawn_index] = "P"
    pieces_fen = "/".join("".join(pieces_fen[beg:beg + 8]) for beg in range(0, 64, 8))
    for ln in range(8, 0, -1):
        pieces_fen = pieces_fen.replace("_" * ln, str(ln))

    res = pieces_fen + " " + fen_ending
    return res


def get_organized_fen_examples():
    """Vraci feny v nejakym lepsim postupu nez uplne random feny"""
    fen1 = "8/8/8/kK6/8/8/8/8 w KQkq - 0 1"


if __name__ == "__main__":
    # for fen generation
    random.seed(42)
    batch_size = 256

    with open("val_fens_simple.txt") as f:
        val_fens = f.read().split("\n")
    with open("val_fens_simple_multi.txt") as f:
        val_fens2 = f.read().split("\n")
    # val_fens = [create_simple_random_fen(piece_count=10) for _ in range(100)]
    # print(*val_fens, sep="\n", end="")
    # exit()

    model = PositionEvaluator(input_size=128+4)
    model.train()

    loss_fn = torch.nn.MSELoss()
    # print(list(model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10_000):
        total_loss = 0
        optimizer.zero_grad()
        for _ in range(batch_size):
            fen = create_simple_random_fen(piece_count=None)
            board = chess.Board(fen)
            position_value = evaluate_position_piece_value(fen)
            vec = fen_to_vec(fen)
            model_eval = model(vec)
            # print(position_value)
            # print(model_eval.item())
            loss = loss_fn(model_eval, torch.tensor([position_value], dtype=torch.float))
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        if not epoch % 10:
            print(total_loss.item())
            val_loss = 0
            with torch.no_grad():
                for fen in val_fens:
                    board = chess.Board(fen)
                    position_value = evaluate_position_piece_value(fen)
                    vec = fen_to_vec(fen)
                    model_eval = model(vec)
                    val_loss += loss_fn(model_eval, torch.tensor([position_value], dtype=torch.float))
            print("Validation:", val_loss.item())
            val_loss = 0
            with torch.no_grad():
                for fen in val_fens2:
                    board = chess.Board(fen)
                    position_value = evaluate_position_piece_value(fen)
                    vec = fen_to_vec(fen)
                    model_eval = model(vec)
                    val_loss += loss_fn(model_eval, torch.tensor([position_value], dtype=torch.float))
            print("Validation:", val_loss.item())
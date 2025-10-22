import random
import chess
import torch
import time

from fen_to_vec import fen_to_vec
from model import NNModel, evaluate_position_piece_value


def create_random_fen(piece_count=None, weights=None):
    """Creates fens of positions that dont necessarily make sense, just for piece value"""
    if piece_count is None:
        piece_count = random.randint(0, 32)
    if piece_count > 62:
        raise ValueError("piece_count too high, max allowed is 62 to leave space for 2 kings")
    fen_ending = "- 0 1"
    # give different castling rights with 50 percent chance
    castling = "".join([char for char in "KQkq" if random.random() > 0.5])
    if not castling:
        castling = "-"
    color_turn = "w" if random.random() > 0.5 else "b"
    fen_ending = color_turn + " " + castling + " " + fen_ending

    pieces = "PNBRQ"
    if weights is None:
        # 8 times higher weight for pawns
        weights = [4] + [1 for _ in pieces[1:]]
        # same weights for black pieces
        weights *= 2
    pieces += pieces.lower()
    # "_" symbolizes empty square
    pieces_fen = ["_" for _ in range(64)]
    # add two squares for kings
    selected_squares = random.sample(range(64), k=piece_count + 2)
    # first place kings (first two selected squares)
    for square, king_char in zip(selected_squares[:2], ("K", "k")):
        # place white and black king
        pieces_fen[square] = king_char
    # add all other pieces
    for square in selected_squares[2:]:
        # use choices becuase of weight param, but leave k=1
        piece = random.choices(pieces, weights=weights, k=1)[0]
        pieces_fen[square] = piece
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("data/val_fens_rand_piece_20000.txt") as f:
        val_fens = f.read().split("\n")
    # with open("data/val_fens_simple_multi.txt") as f:
    #     val_fens2 = f.read().split("\n")
    # val_fens = [create_random_fen(piece_count=None) for _ in range(20_000)]
    # with open("data/val_fens_rand_piece_20000.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(val_fens))
    # exit()

    piece_count = None
    learning_rate = 0.0001

    model = NNModel().to(device)
    model.load_state_dict(torch.load("weights/3_piece.pth", map_location=device))

    model.train()

    loss_fn = torch.nn.MSELoss().to(device)
    # print(list(model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_position_values = torch.tensor([evaluate_position_piece_value(fen) for fen in val_fens], dtype=torch.float).unsqueeze(1).to(device)
    val_vecs = torch.stack([fen_to_vec(fen) for fen in val_fens]).to(device)

    epochs = 200
    batches_per_epoch = 1_000

    for epoch in range(epochs):
        total_loss = 0
        for batch_number in range(batches_per_epoch):
            optimizer.zero_grad()

            fens = [create_random_fen(piece_count=piece_count) for _ in range(batch_size)]
            # boards = [chess.Board(fen) for fen in fens]
            position_values = torch.tensor([evaluate_position_piece_value(fen) for fen in fens], dtype=torch.float).unsqueeze(1).to(device)
            vecs = torch.stack([fen_to_vec(fen) for fen in fens]).to(device)
            model_evals = model(vecs)
            loss = loss_fn(model_evals, position_values)
            loss.backward()
            optimizer.step()
            # just for monitoring
            total_loss += loss

        # for _ in range(batch_size):
        #     beg = time.perf_counter()
        #     fen = create_random_fen(piece_count=1)
        #     end = time.perf_counter()
        #     print(f"Creating fen took {end - beg:.5f}")

        #     beg = time.perf_counter()
        #     # board = chess.Board(fen)

        #     end = time.perf_counter()
        #     print(f"Creating board took {end - beg:.5f}")

        #     beg = time.perf_counter()
        #     position_value = evaluate_position_piece_value(fen)
        #     end = time.perf_counter()
        #     print(f"Value by piece took {end - beg:.5f}")

        #     beg = time.perf_counter()
        #     vec = fen_to_vec(fen).to(device)
        #     end = time.perf_counter()
        #     print(f"fen2vec took {end - beg:.5f}")

        #     beg = time.perf_counter()
        #     model_eval = model(vec)
        #     end = time.perf_counter()
        #     print(f"Model took {end - beg:.5f}")
        #     # print(position_value)
        #     # print(model_eval.item())
        #     beg = time.perf_counter()
        #     loss = loss_fn(model_eval, torch.tensor([position_value], dtype=torch.float).to(device))
        #     end = time.perf_counter()
        #     print(f"Loss took {end - beg:.5f}")
        #     total_loss += loss

        # total_loss.backward()
        # optimizer.step()

        print(f"Epoch {epoch}:")
        print(total_loss.item() / batches_per_epoch)
        with torch.no_grad():
            val_model_evals = model(val_vecs)
            val_loss = loss_fn(val_model_evals, val_position_values)
        print("Validation:", val_loss.item())

        torch.save(model.state_dict(), "weights/model.pth")
            # val_loss = 0
            # with torch.no_grad():
            #     for fen in val_fens2:
            #         board = chess.Board(fen)
            #         position_value = evaluate_position_piece_value(fen)
            #         vec = fen_to_vec(fen)
            #         model_eval = model(vec)
            #         val_loss += loss_fn(model_eval, torch.tensor([position_value], dtype=torch.float))
            # print("Validation:", val_loss.item())
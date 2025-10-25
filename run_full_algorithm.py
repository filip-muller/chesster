from model import find_best_move

if __name__ == "__main__":
    while True:
        fen = input("fen: ").strip()
        best_move = find_best_move(fen, depth=3)
        print(best_move)

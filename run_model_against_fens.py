import torch

from model import NNModel
from fen_to_vec import fen_to_vec

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NNModel().to(device)
    model.eval()
    model.load_state_dict(torch.load("weights/full_pieces.pth", map_location=device))
    while True:
        fen = input("fen: ").strip()
        vec = fen_to_vec(fen).to(device)
        with torch.no_grad():
            eval = model(vec).item()
        print(eval)

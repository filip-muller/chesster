"""Converts weights from versions using different fen2vec functions"""

import torch

# From color encoding with a "color to move" feature
# Original: 64 (piece, color) combos = 128, 4 castling (color), 1 color to move (133 total)
# New: 64 (piece, is_color_to_move) combos = 128, 4 castling (is_color_to_move) (132 total)
# We keep the value that used to mean WHITE to now mean COLOR_TO_MOVE and just remove the
# weights related to the last input (used to be color to move)
def v0_to_color_to_move(state_dict):
    out = state_dict.copy()
    first_layer_weights = out["fc_layers.0.weight"]
    out["fc_layers.0.weight"] = torch.cat((first_layer_weights[:, :128], first_layer_weights[:, 129:]), dim=1)
    return out


if __name__ == "__main__":
    old = torch.load("weights/900_plus_1500.pth", map_location="cpu")
    new = v0_to_color_to_move(old)
    torch.save(new, "weights/new/900_plus_1500.pth")

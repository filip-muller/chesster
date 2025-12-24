import unittest
import torch

import fen_to_vec


class TestFenToVec(unittest.TestCase):
    starting_pieces = [fen_to_vec.ROOK, fen_to_vec.KNIGHT, fen_to_vec.BISHOP, fen_to_vec.QUEEN, fen_to_vec.KING, fen_to_vec.BISHOP, fen_to_vec.KNIGHT, fen_to_vec.ROOK]
    row_of_pawns = [fen_to_vec.PAWN] * 8
    empty_row = [fen_to_vec.EMPTY] * 8
    onehot_king = [0, 0, 0, 0, 0, 1]
    onehot_pawn = [1, 0, 0, 0, 0, 0]
    onehot_to_move = [1, 0]
    onehot_not_to_move = [0, 1]

    @staticmethod
    def _apply_color(pieces, color_val):
        """Returns a list like [pieces[0], color_val, pieces[1], color_val, ...]"""
        return [e for piece in pieces for e in (piece, color_val)]

    def test_starting(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = fen_to_vec.fen_to_vec(fen)

        expected = self._apply_color(self.starting_pieces, fen_to_vec.COLOR_NOT_TO_MOVE)
        expected += self._apply_color(self.row_of_pawns, fen_to_vec.COLOR_NOT_TO_MOVE)
        expected += 4 * self._apply_color(self.empty_row, fen_to_vec.NEUTRAL)
        expected += self._apply_color(self.row_of_pawns, fen_to_vec.COLOR_TO_MOVE)
        expected += self._apply_color(self.starting_pieces, fen_to_vec.COLOR_TO_MOVE)

        # castling
        expected += [fen_to_vec.CAN_CASTLE] * 4

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))

    def test_starting_black_to_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        res = fen_to_vec.fen_to_vec(fen)

        expected = self._apply_color(self.starting_pieces, fen_to_vec.COLOR_TO_MOVE)
        expected += self._apply_color(self.row_of_pawns, fen_to_vec.COLOR_TO_MOVE)
        expected += 4 * self._apply_color(self.empty_row, fen_to_vec.NEUTRAL)
        expected += self._apply_color(self.row_of_pawns, fen_to_vec.COLOR_NOT_TO_MOVE)
        expected += self._apply_color(self.starting_pieces, fen_to_vec.COLOR_NOT_TO_MOVE)

        # castling
        expected += [fen_to_vec.CAN_CASTLE] * 4

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))

    def test_castling(self):
        fen = "8/8/8/8/8/8/8/8 b Qkq - 0 1"
        res = fen_to_vec.fen_to_vec(fen)

        expected = 8 * self._apply_color(self.empty_row, fen_to_vec.NEUTRAL)

        # castling
        expected += [fen_to_vec.CAN_CASTLE] * 2 + [fen_to_vec.CANNOT_CASTLE] + [fen_to_vec.CAN_CASTLE]

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))

    def test_onehot_simple(self):
        fen = "k7/8/8/8/8/8/8/6K1 w KQkq - 0 1"
        res = fen_to_vec.fen_to_onehot(fen)

        expected = [self.onehot_king + self.onehot_not_to_move + [0]]
        expected += [self.onehot_king + self.onehot_to_move + [62]]

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))

    def test_onehot_simplified(self):
        fen = "k7/pppppppp/8/8/8/8/PPPPPPPP/6K1 w KQkq - 0 1"
        res = fen_to_vec.fen_to_onehot(fen)

        expected = [self.onehot_king + self.onehot_not_to_move + [0]]
        white_pawn = self.onehot_pawn + self.onehot_not_to_move
        expected += [white_pawn + [pos] for pos in range(8, 16)]

        black_pawn = self.onehot_pawn + self.onehot_to_move
        expected += [black_pawn + [pos] for pos in range(48, 56)]

        expected += [self.onehot_king + self.onehot_to_move + [62]]

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))

    def test_onehot_simplified_black(self):
        fen = "k7/pppppppp/8/8/8/8/PPPPPPPP/6K1 b KQkq - 0 1"
        res = fen_to_vec.fen_to_onehot(fen)

        expected = [self.onehot_king + self.onehot_to_move + [0]]
        white_pawn = self.onehot_pawn + self.onehot_to_move
        expected += [white_pawn + [pos] for pos in range(8, 16)]

        black_pawn = self.onehot_pawn + self.onehot_not_to_move
        expected += [black_pawn + [pos] for pos in range(48, 56)]

        expected += [self.onehot_king + self.onehot_not_to_move + [62]]

        expected = torch.Tensor(expected)
        self.assertTrue(torch.allclose(res, expected))


if __name__ == "__main__":
    unittest.main()

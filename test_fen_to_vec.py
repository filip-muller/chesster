import unittest
import torch

import fen_to_vec


class TestFenToVec(unittest.TestCase):
    starting_pieces = [fen_to_vec.ROOK, fen_to_vec.KNIGHT, fen_to_vec.BISHOP, fen_to_vec.QUEEN, fen_to_vec.KING, fen_to_vec.BISHOP, fen_to_vec.KNIGHT, fen_to_vec.ROOK]
    row_of_pawns = [fen_to_vec.PAWN] * 8
    empty_row = [fen_to_vec.EMPTY] * 8

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


if __name__ == "__main__":
    unittest.main()

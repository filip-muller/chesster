import unittest

from fen_to_vec import fen_to_vec


class TestFenToVec(unittest.TestCase):
    def test_starting(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = fen_to_vec(fen)
        print(res)

        expected = [104, 102, 103, 105, 106, 103, 102, 104] + [101] * 8 + [0] * 32 + [1] * 8 + [4, 2, 3, 5, 6, 3, 2, 4]
        # castling
        expected += [1, 1, 1, 1]

        self.assertEqual(res, expected)


if __name__ == "__main__":
    unittest.main()

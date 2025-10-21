import unittest

from model import evaluate_position_piece_value


class TestModel(unittest.TestCase):
    STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_evaluate_position_piece_value(self):
        res = evaluate_position_piece_value(self.STARTING_FEN)
        self.assertEqual(res, 0)

        fen = "n" + self.STARTING_FEN[1:]
        res = evaluate_position_piece_value(fen)
        self.assertEqual(res, 2)

        fen = self.STARTING_FEN.replace("8", "B7")
        res = evaluate_position_piece_value(fen)
        self.assertEqual(res, 12)

        fen = self.STARTING_FEN.replace("8", "r7")
        res = evaluate_position_piece_value(fen)
        self.assertEqual(res, -20)

        fen = "8/8/8/kK6/8/8/8/8 w KQkq - 0 1"
        res = evaluate_position_piece_value(fen)
        self.assertEqual(res, 0)

        fen = "p7/8/8/kK6/8/8/8/8 w KQkq - 0 1"
        res = evaluate_position_piece_value(fen)
        self.assertEqual(res, -1)



if __name__ == "__main__":
    unittest.main()

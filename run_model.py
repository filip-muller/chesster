from model import PositionEvaluator, min_max_eval


fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
evaluator = PositionEvaluator()

print(evaluator.evaluate_positions(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"])[0])

print(min_max_eval(evaluator, fen))

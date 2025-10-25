from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chess

from model import find_best_move


app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Import your evaluator and min_max_eval function here
# from your_module import Evaluator, min_max_eval

# Initialize your evaluator
# evaluator = Evaluator(model_path="your_model.pth")

@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.json
        fen = data.get('fen')
        depth = data.get('depth', 2)

        if not fen:
            return jsonify({'error': 'No FEN provided'}), 400

        # Validate FEN
        try:
            board = chess.Board(fen)
        except ValueError:
            return jsonify({'error': 'Invalid FEN'}), 400

        # Check if game is over
        if board.is_game_over():
            return jsonify({'error': 'Game is over'}), 400

        # Get the best move using your minimax function
        # best_move, eval_score = min_max_eval(evaluator, fen, depth=depth)

        # For testing without the actual model, use a random move:
        # import random
        # legal_moves = list(board.legal_moves)
        # best_move = random.choice(legal_moves)
        eval_score = 0.0
        best_move = find_best_move(fen, depth)

        # # Convert move to UCI format (e.g., "e2e4")
        move_uci = best_move.uci()
        print(move_uci)

        return jsonify({
            'move': move_uci,
            'evaluation': float(eval_score),
            'depth': depth
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route("/")
def home():
    # with open("html/index.html") as f:
    #     return f.read()
    return send_from_directory("html/", "index.html")

@app.route('/img/<path:path>')
def serve_static(path):
    return send_from_directory('html/img', path)



if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

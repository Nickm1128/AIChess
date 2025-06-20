import random
from typing import Dict, List, Tuple
import os
import pickle
import numpy as np
import chess
import chess.engine

from Agent import Agent, Synapse
from mutations import mutate_agent

SAVE_DIRECTORY = "trained_agents"
# Path to the Stockfish engine. The repository previously expected the engine
# to be located in ``/workspace/stockfish`` but that binary is not included.
# Use the system wide Stockfish installation if available.
ENGINE_PATH = '/workspace/stockfish/stockfish-ubuntu-x86-64-avx2'


def encode_board(board: chess.Board) -> List[float]:
    """Encode a board into a simple list of floats."""
    mapping = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    data = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            data.append(0.0)
        else:
            val = mapping[piece.piece_type] / 6.0
            data.append(val if piece.color == chess.WHITE else -val)
    data.append(1.0 if board.turn == chess.WHITE else -1.0)
    return data


def choose_move(agent: Agent, board: chess.Board) -> chess.Move:
    """Select a legal move based on the agent's output neurons.

    This function now guarantees the returned move is legal. If, for any
    unexpected reason, the sampled move is not legal, it falls back to a random
    legal move.
    """

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    outputs = np.array([n.state for n in agent.output_neurons])

    # Resize outputs to the number of legal moves
    if len(outputs) < len(legal_moves):
        outputs = np.pad(outputs, (0, len(legal_moves) - len(outputs)), constant_values=0)
    elif len(outputs) > len(legal_moves):
        outputs = outputs[:len(legal_moves)]

    exp_outputs = np.exp(outputs - np.max(outputs))  # Softmax (stable)
    probs = exp_outputs / np.sum(exp_outputs)

    index = np.random.choice(len(legal_moves), p=probs)
    move = legal_moves[index]

    # Extra safety: ensure the move is legal (should always be true)
    if move not in board.legal_moves:
        move = random.choice(legal_moves)

    return move


def generate_batch(engine: chess.engine.SimpleEngine, batch_size: int, depth: int = 1
                   ) -> Tuple[List[chess.Board], List[chess.Move]]:
    """Generate a batch of board positions and Stockfish's chosen moves."""
    boards = []
    moves = []
    board = chess.Board()
    for _ in range(batch_size):
        if board.is_game_over():
            board = chess.Board()
        result = engine.play(board, chess.engine.Limit(depth=depth))
        boards.append(board.copy())
        moves.append(result.move)
        board.push(result.move)
    return boards, moves


def generate_cached_positions(
    engine: chess.engine.SimpleEngine, count: int, depth: int = 1
) -> List[Dict[str, object]]:
    """Precompute evaluations for a number of board positions."""
    print('Generating position cache...')
    cache = []
    board = chess.Board()
    for _ in range(count):
        if _ % 10 == 0:
            print(f'Completed {_} generated positions')
        if board.is_game_over():
            board = chess.Board()

        # Determine Stockfish best move for the current position
        result = engine.play(board, chess.engine.Limit(depth=depth))
        best_move = result.move

        evaluations = {}
        for move in board.legal_moves:
            board_after = board.copy()
            board_after.push(move)
            info = engine.analyse(board_after, chess.engine.Limit(depth=depth))
            evaluations[move.uci()] = info["score"].white().score(mate_score=100000)

        cache.append(
            {
                "fen": board.fen(),
                "best_move": best_move.uci(),
                "evals": evaluations,
            }
        )

        board.push(best_move)

    return cache


def evaluate_agent(
    agent: Agent,
    engine: chess.engine.SimpleEngine,
    boards: List[chess.Board],
    moves: List[chess.Move],
    depth: int = 1,
) -> float:
    """Return a score based on how close the agent's moves are to Stockfish.

    Instead of simply counting correct predictions, this function analyses the
    board evaluation after the agent's chosen move and compares it with the
    evaluation after Stockfish's move. The closer the two evaluations are, the
    higher (less negative) the returned score will be.
    """

    score = 0.0
    for board, true_move in zip(boards, moves):
        agent.reset()
        inputs = encode_board(board)
        agent.receive_inputs(inputs)
        agent.step(think=5)

        predicted = choose_move(agent, board)

        # Evaluate Stockfish's chosen continuation
        board_true = board.copy()
        board_true.push(true_move)
        info_true = engine.analyse(board_true, chess.engine.Limit(depth=depth))
        stockfish_eval = info_true["score"].white().score(mate_score=100000)

        # Evaluate the agent's chosen continuation
        board_pred = board.copy()
        if predicted is not None:
            board_pred.push(predicted)
            info_pred = engine.analyse(
                board_pred, chess.engine.Limit(depth=depth)
            )
            agent_eval = info_pred["score"].white().score(mate_score=100000)
        else:
            # No legal moves - heavily penalise
            agent_eval = -100000

        score -= abs(agent_eval - stockfish_eval)

    return score


def evaluate_cached(agent: Agent, entry: Dict[str, object]) -> Tuple[float, str]:
    """Evaluate agent on a cached board position."""
    board = chess.Board(entry["fen"])
    agent.reset()
    inputs = encode_board(board)
    agent.receive_inputs(inputs)
    agent.step(think=5)

    move = choose_move(agent, board)
    if move is None:
        return -100000.0, ""

    move_uci = move.uci()
    pred_eval = entry["evals"].get(move_uci, -100000)
    best_eval = entry["evals"][entry["best_move"]]

    score = -abs(pred_eval - best_eval)
    return score, move_uci


def create_agent(name: str = 'Agent', neuron_count: int = 10) -> Agent:
    """Create an agent with a custom number of synapses per neuron."""
    agent = Agent(name, neuron_count)
    agent.synapses = []
    for _ in range(neuron_count):
        for __ in range(random.randint(7, 12)):
            pre, post = random.sample(agent.neurons, 2)
            weight = random.uniform(-1, 1)
            agent.synapses.append(Synapse(pre, post, weight))
    return agent


def train_cached(
    positions: int = 1000,
    depth: int = 1,
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.2,
    attempts_per_batch: int = 500,
    rounds: int = 1_000,
) -> Agent:
    """Train using cached Stockfish evaluations.

    Each round generates ``positions`` board states. The agent is scored on the
    sum of its evaluations across all of these states. ``attempts_per_batch``
    controls how many mutations are tried for the entire batch of positions.
    """

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    agent = create_agent("StockfishMimic", 10)

    best_score_tracker = []

    for _ in range(rounds):
        # Generate a fresh cache each round so the agent sees new positions
        cache = generate_cached_positions(engine, positions, depth)

        def evaluate_batch(test_agent: Agent) -> float:
            total = 0.0
            for entry in cache:
                score, _ = evaluate_cached(test_agent, entry)
                total += score
            return total

        parent_score = evaluate_batch(agent)
        best_agent = agent
        best_score = parent_score

        for _m in range(attempts_per_batch):
            mutant = mutate_agent(agent, mutation_rate, mutation_strength)
            score = evaluate_batch(mutant)
            if score > best_score:
                best_score = score
                best_agent = mutant

        best_score_tracker.append(best_score)

        if best_score > parent_score:
            agent = best_agent

        print(f"Best agent neuron count: {len(agent.neurons)}")
        print(f"Best agent synapse count: {len(agent.synapses)}")        

    # Ensure the save directory exists before writing any files
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    save_path = os.path.join(SAVE_DIRECTORY, "best_agent.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(agent, f)
    print(f"Saved agent to {save_path}")

    save_path = os.path.join(SAVE_DIRECTORY, "best_scores.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(best_score_tracker, f)
    print(f"Saved scores record to {save_path}")

    engine.quit()
    return agent


def train(rounds: int = 1_000_000, batch_size: int = 32, depth: int = 1,
          mutation_rate: float = 0.9, mutation_strength: float = 1.0) -> Agent:
    mm = 1
    best_score_round = 0
    """Evolve an agent to mimic Stockfish move choices."""
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    agent = create_agent('StockfishMimic', 10)
    boards, moves = generate_batch(engine, batch_size, depth)
    best_score = evaluate_agent(agent, engine, boards, moves, depth)

    for r in range(rounds):
        multiplier = np.random.uniform(0,.1)
            
        mutant = mutate_agent(agent, mutation_rate * multiplier, mutation_strength * multiplier)
        score = evaluate_agent(mutant, engine, boards, moves, depth)
        if score > best_score:
            agent = mutant
            best_score = score
            best_score_round = r
            #boards, moves = generate_batch(engine, batch_size, depth)
        if r % 5 == 0:
            boards, moves = generate_batch(engine, batch_size, depth)
            best_score = evaluate_agent(agent, engine, boards, moves, depth)
        if r % 100 == 0:
            print(f'Round {r}: best score {best_score}/{batch_size}')

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    save_path = os.path.join(SAVE_DIRECTORY, "best_agent.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(agent, f)
    print(f"Saved agent to {save_path}")

    engine.quit()
    return agent


if __name__ == '__main__':
    train_cached()

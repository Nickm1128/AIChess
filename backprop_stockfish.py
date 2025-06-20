import random
from typing import List, Tuple
import os
import pickle

import chess
import chess.engine

from Agent import Agent, Synapse
from mutations import mutate_agent

SAVE_DIRECTORY = "trained_agents"
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
    """Select a legal move based on the agent's output neurons."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    outputs = np.array([n.state for n in agent.output_neurons])
    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    index = np.random.choice(len(legal_moves), p=probs / np.sum(probs))
    return legal_moves[index]


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


def evaluate_agent(agent: Agent, boards: List[chess.Board], moves: List[chess.Move]) -> int:
    """Return how many moves the agent predicts correctly for the given boards."""
    correct = 0
    for board, true_move in zip(boards, moves):
        agent.reset()
        inputs = encode_board(board)
        agent.receive_inputs(inputs)
        agent.step(think=5)
        predicted = choose_move(agent, board)
        if predicted == true_move:
            correct += 1
    return correct


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


def train(rounds: int = 1_000_000, batch_size: int = 32, depth: int = 1,
          mutation_rate: float = 0.9, mutation_strength: float = 0.9) -> Agent:
    """Evolve an agent to mimic Stockfish move choices."""
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    agent = create_agent('StockfishMimic', 10)
    boards, moves = generate_batch(engine, batch_size, depth)
    best_score = evaluate_agent(agent, boards, moves)

    for r in range(rounds):
        multiplier = (rounds -  r) / rounds
        multiplier = max(0.2, (rounds - r) / rounds)
        
        mutant = mutate_agent(agent, mutation_rate * multiplier, mutation_strength * multiplier)
        score = evaluate_agent(mutant, boards, moves)
        if score > best_score:
            agent = mutant
            best_score = score
            boards, moves = generate_batch(engine, batch_size, depth)
        if r % 5 == 0:
            boards, moves = generate_batch(engine, batch_size, depth)
            best_score = evaluate_agent(agent, boards, moves)
        if r % 100 == 0:
            print(f'Round {r}: best score {best_score}/{batch_size}')

    save_path = os.path.join(SAVE_DIRECTORY, f"best_agent.pkl")
    if len(population) > 0:
        with open(save_path, 'wb') as f:
            pickle.dump(population, f)
        print(f"Saved agent to {save_path}")

    engine.quit()
    return agent


if __name__ == '__main__':
    train()

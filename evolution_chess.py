import chess
import random
import numpy as np

from Agent import Agent
from mutations import mutate_agent


def encode_board(board):
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
            val = mapping[piece.piece_type]
            val = val / 6.0
            if piece.color == chess.WHITE:
                data.append(val)
            else:
                data.append(-val)
    data.append(1.0 if board.turn == chess.WHITE else -1.0)
    return data


def choose_move(agent, board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    output_val = np.mean([n.state for n in agent.output_neurons])
    index = int(((output_val + 1) / 2) * len(legal_moves))
    index = max(0, min(index, len(legal_moves) - 1))
    return legal_moves[index]


def play_game(agent_white, agent_black, max_moves=40):
    board = chess.Board()
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        agent = agent_white if board.turn == chess.WHITE else agent_black
        agent.reset()
        inputs = encode_board(board)
        agent.receive_inputs(inputs)
        agent.step(think=2)
        move = choose_move(agent, board)
        if move is None or move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        move_count += 1
    result = board.result(claim_draw=True)
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0


def evaluate_agent(agent, games=2):
    """Evaluate an agent against a random opponent."""
    score = 0
    for _ in range(games):
        random_opponent = RandomAgent(len(agent.neurons))
        score += play_game(agent, random_opponent)
        score -= play_game(random_opponent, agent)
    return score


def evaluate_match(agent_a, agent_b, games=2):
    """Play agents against each other and return the score for agent_a."""
    score = 0
    for _ in range(games):
        score += play_game(agent_a, agent_b)
        score -= play_game(agent_b, agent_a)
    return score


class RandomAgent(Agent):
    def __init__(self, neuron_count):
        super().__init__('Random', neuron_count)

    def step(self, think=1):
        pass  # random agent does not use network

    def receive_inputs(self, inputs):
        pass

    def decide_move(self, board):
        return random.choice(list(board.legal_moves))


def competitive_evolution(agent_a, agent_b, rounds=10, attempts=5,
                          mutation_rate=0.1, mutation_strength=0.2):
    """Evolve two agents by alternately mutating them until a mutation wins."""
    for r in range(rounds):
        improved_a = False
        for _ in range(attempts):
            mutant = mutate_agent(agent_a, mutation_rate, mutation_strength)
            if evaluate_match(mutant, agent_b) > 0:
                agent_a = mutant
                improved_a = True
                break

        improved_b = False
        for _ in range(attempts):
            mutant = mutate_agent(agent_b, mutation_rate, mutation_strength)
            if evaluate_match(mutant, agent_a) > 0:
                agent_b = mutant
                improved_b = True
                break

        print(f"Round {r}: A improved={improved_a}, B improved={improved_b}")
    return agent_a, agent_b


if __name__ == '__main__':
    neuron_count = 70
    agent_a = Agent('AgentA', neuron_count)
    agent_b = Agent('AgentB', neuron_count)

    # Run competitive evolution between the two agents
    agent_a, agent_b = competitive_evolution(agent_a, agent_b, rounds=10, attempts=5)

    print('Final duel score:', evaluate_match(agent_a, agent_b))

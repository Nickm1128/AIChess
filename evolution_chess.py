import chess
import chess.engine
import chess.svg
import random
import numpy as np
import pickle
import copy
import imageio
import cairosvg
import io

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
        agent.think_until_convergence()
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
    """Play agents against each other and return score and win counts."""
    score = 0
    wins_a = 0
    wins_b = 0
    for _ in range(games):
        result = play_game(agent_a, agent_b)
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        score += result

        result = play_game(agent_b, agent_a)
        if result == 1:
            wins_b += 1
        elif result == -1:
            wins_a += 1
        score -= result  # subtract because perspective flips

    return score, wins_a, wins_b


def evaluate_board(board, engine, depth=1):
    """Return Stockfish's evaluation in centipawns from White's perspective."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=10000)


def play_game_vs_stockfish(agent, engine, play_as_white=True, max_moves=40, depth=1, round=0):
    """Play a single game against Stockfish and return the result from the agent's perspective."""
    board = chess.Board()
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == (chess.WHITE if play_as_white else chess.BLACK):
            # Evaluate board before the agent moves
            before = evaluate_board(board, engine, depth)
            if not play_as_white:
                before = -before

            agent.reset()
            inputs = encode_board(board)
            agent.receive_inputs(inputs)
            agent.think_until_convergence()
            move = choose_move(agent, board)
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))
            board.push(move)

            after = evaluate_board(board, engine, depth)
            if not play_as_white:
                after = -after

            # Reward is improvement in evaluation scaled to roughly [-1,1]
            reward = (after - before) / 100.0
            agent.learn(reward)
        else:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            board.push(result.move)
        move_count += 1

    result = board.result(claim_draw=True)
    if result == '1-0':
        final_reward = 1 if play_as_white else -1
    elif result == '0-1':
        final_reward = -1 if play_as_white else 1
    else:
        final_reward = 0

    if np.random.rand() > .99:
        if final_reward == 1:
            print(f'Round: {round} Agent beat stockfish, depth {depth}. Neurons: {len(agent.neurons)}')
        elif final_reward == -1:
            print(f'Round: {round} Agent lost to stockfish, depth {depth}. Neurons: {len(agent.neurons)}')
        else:
            print(f'Round: {round} Agent tied with stockfish, depth {depth}. Neurons: {len(agent.neurons)}')
    agent.learn(final_reward)
    return final_reward


def record_game_vs_stockfish(agent, engine_path, gif_path,
                              play_as_white=True, max_moves=40, depth=1):
    """Play a game against Stockfish and save the moves as an animated GIF."""
    print('Recording stockfish game')
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board()
    frames = []
    move_count = 0
    gif_path = 'workspace/AIChess/' + gif_path
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == (chess.WHITE if play_as_white else chess.BLACK):
            agent.reset()
            inputs = encode_board(board)
            agent.receive_inputs(inputs)
            agent.think_until_convergence()
            move = choose_move(agent, board)
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))
            board.push(move)
        else:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            board.push(result.move)

        svg_data = chess.svg.board(board=board).encode("utf-8")
        png_data = cairosvg.svg2png(bytestring=svg_data)
        frames.append(imageio.v2.imread(io.BytesIO(png_data)))
        move_count += 1

    engine.quit()
    if frames:
        imageio.mimsave(gif_path, frames, duration=0.5)


def pretrain_agent(
    agent,
    engine_path: str = '/workspace/stockfish/stockfish-ubuntu-x86-64-avx2',
    games: int = 50,
    depth: int = 1,
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.2,
):
    """Warm start an agent using Stockfish evaluations for feedback.

    The agent is periodically mutated during pretraining. After each pair of
    games against Stockfish, a mutated copy of the agent is created. If the
    mutated version achieves a better or equal result against Stockfish than the
    current agent, the mutation is kept for subsequent rounds.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    multiplier = 1
    for _ in range(games):
        base_score = 0
        base_score += play_game_vs_stockfish(agent, engine, play_as_white=True, depth=depth, round = _)
        base_score += play_game_vs_stockfish(agent, engine, play_as_white=False, depth=depth, round = _)

        # Attempt a mutation and keep it if performance does not decrease
        mutant = mutate_agent(agent, mutation_rate * multiplier, mutation_strength * multiplier)
        multiplier = (games - _) / games
        mutant_score = 0
        mutant_score += play_game_vs_stockfish(mutant, engine, play_as_white=True, depth=depth, round = _)
        mutant_score += play_game_vs_stockfish(mutant, engine, play_as_white=False, depth=depth, round = _)

        if mutant_score >= base_score:
            agent = mutant

        if mutant_score > 1:
            depth = np.min([depth+1, 10])
    engine.quit()
    return agent


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
                          mutation_rate=0.1, mutation_strength=0.2,
                          max_skill_gap=5, pretrain_games=0,
                          stockfish_path = '/workspace/stockfish/stockfish-ubuntu-x86-64-avx2'):
    """Evolve two agents while keeping their skill levels relatively close."""

    if pretrain_games > 0:
        print(f"Pretraining agents for {pretrain_games} games against Stockfish...")
        agent_a = pretrain_agent(agent_a, stockfish_path, games=pretrain_games)
        #agent_b = pretrain_agent(agent_b, stockfish_path, games=pretrain_games)
    agent_b = agent_a.copy()
    for r in range(rounds):
        skill_a = evaluate_agent(agent_a)
        skill_b = evaluate_agent(agent_b)
        skill_diff = skill_a - skill_b

        allow_a = True
        allow_b = True
        attempts_a = attempts
        attempts_b = attempts

        if skill_diff > max_skill_gap:
            # Agent A is too far ahead; pause its improvement and give B more chances
            allow_a = False
            attempts_b *= 2
        elif -skill_diff > max_skill_gap:
            # Agent B is too far ahead
            allow_b = False
            attempts_a *= 2

        improved_a = False
        if allow_a:
            for _ in range(attempts_a):
                mutant = mutate_agent(agent_a, mutation_rate, mutation_strength)
                score, _, _ = evaluate_match(mutant, agent_b)
                if score > 0:
                    agent_a = mutant
                    improved_a = True
                    break

        improved_b = False
        if allow_b:
            for _ in range(attempts_b):
                mutant = mutate_agent(agent_b, mutation_rate, mutation_strength)
                score, _, _ = evaluate_match(mutant, agent_a)
                if score > 0:
                    agent_b = mutant
                    improved_b = True
                    break

        match_score, wins_a, wins_b = evaluate_match(agent_a, agent_b)
        skill_a = evaluate_agent(agent_a)
        skill_b = evaluate_agent(agent_b)

        print(f"Round {r}: A improved={improved_a}, B improved={improved_b}")
        print(
            f"    AgentA - skill {skill_a}, neurons {len(agent_a.neurons)}, wins vs B {wins_a}"
        )
        print(
            f"    AgentB - skill {skill_b}, neurons {len(agent_b.neurons)}, wins vs A {wins_b}"
        )

        if (r + 1) in {100,  10_000, 100_000, 500_000}:
            gif_name = f"round_{r + 1}_agent_vs_stockfish.gif"
            record_game_vs_stockfish(agent_a, stockfish_path, gif_name)
    return agent_a, agent_b


if __name__ == '__main__':
    neuron_count = 10
    agent_a = Agent('AgentA', neuron_count)
    agent_b = Agent('AgentB', neuron_count)

    # Run competitive evolution between the two agents. Games against Stockfish
    # will be rendered at 10k, 100k and 500k rounds.
    agent_a, agent_b = competitive_evolution(
        agent_a,
        agent_b,
        rounds=500_000,
        attempts=100,
        pretrain_games=1_000_000,
    )

    # Save the evolved agents
    with open('agent_a.pkl', 'wb') as f_a:
        pickle.dump(agent_a, f_a)
    with open('agent_b.pkl', 'wb') as f_b:
        pickle.dump(agent_b, f_b)

    final_score, wins_a, wins_b = evaluate_match(agent_a, agent_b)
    print('Final duel score:', final_score)
    print(f'AgentA wins: {wins_a}, AgentB wins: {wins_b}')

import chess
import chess.engine
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def encode_board(board: chess.Board):
    """Encode a board into a vector for the neural network."""
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
            if piece.color == chess.WHITE:
                data.append(val)
            else:
                data.append(-val)
    data.append(1.0 if board.turn == chess.WHITE else -1.0)
    return data

ENGINE_PATH = '/usr/games/stockfish'

MOVE_SPACE = 64 * 64  # from_square * 64 + to_square, ignore promotions


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


def generate_dataset(games: int = 10, depth: int = 1, max_moves: int = 40):
    """Generate (state, move) pairs from Stockfish self-play."""
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    states = []
    moves = []
    for _ in range(games):
        board = chess.Board()
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            move = result.move
            states.append(encode_board(board))
            moves.append(move_to_index(move))
            board.push(move)
            move_count += 1
    engine.quit()
    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(moves, dtype=torch.long)
    return X, y


class NeuralAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, MOVE_SPACE)
        )

    def forward(self, x):
        return self.net(x)

    def predict_move(self, board: chess.Board):
        self.eval()
        with torch.no_grad():
            inp = torch.tensor([encode_board(board)], dtype=torch.float32)
            logits = self(inp)[0]
            legal = list(board.legal_moves)
            if not legal:
                return None
            indices = [move_to_index(m) for m in legal]
            legal_logits = logits[indices]
            best = torch.argmax(legal_logits).item()
            return legal[best]


def train(agent: NeuralAgent, X: torch.Tensor, y: torch.Tensor, epochs: int = 5, batch_size: int = 32, lr: float = 0.001):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0
        correct = 0
        total_loss = 0.0
        for batch_x, batch_y in loader:
            opt.zero_grad()
            out = agent(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
        print(f"Epoch {epoch+1}: loss {total_loss/total:.4f}, accuracy {correct/total:.4f}")


def main():
    print("Generating dataset from Stockfish...")
    X, y = generate_dataset(games=20, depth=2)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    agent = NeuralAgent()
    print("Training agent via backprop...")
    train(agent, X_train, y_train, epochs=10, batch_size=64, lr=0.001)
    # Evaluate
    with torch.no_grad():
        out = agent(X_test)
        preds = out.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    print(f"Test accuracy (move prediction): {accuracy:.4f}")

if __name__ == '__main__':
    main()

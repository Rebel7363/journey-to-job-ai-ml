import torch
import torch.nn as nn


class ManualRNNCell(nn.Module):
    """
    Implements a single Vanilla RNN step from mathematical scratch:
    h_t = tanh(x_t @ W_ih.T + b_ih + h_(t-1) @ W_hh.T + b_hh)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input-to-hidden and Hidden-to-hidden linear projections
        self.w_ih = nn.Linear(input_dim, hidden_dim)
        self.w_hh = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_t shape: (batch_size, input_dim)
        # h_prev shape: (batch_size, hidden_dim)
        h_t = self.activation(self.w_ih(x_t) + self.w_hh(h_prev))
        return h_t


class SequenceEmbeddingRNN(nn.Module):
    """
    Processes token sequence through an Embedding layer followed by LSTM.
    Input:  (batch_size, seq_len)
    Output: (batch_size, hidden_dim) final hidden state
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, token_ids: torch.Tensor):
        # 1. Embedding lookup: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embeds = self.embedding(token_ids)

        # 2. LSTM forward pass:
        # out: (batch_size, seq_len, hidden_dim)
        # h_n, c_n: (num_layers, batch_size, hidden_dim)
        out, (h_n, c_n) = self.lstm(embeds)

        # 3. Take last hidden state of the top layer
        final_state = h_n[-1]
        projected = self.fc(final_state)
        return out, projected


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Running Sequence Modeling on Device: {device} ---")

    # 1. Test Manual RNN Cell Step
    batch_size = 2
    embed_dim = 8
    hidden_dim = 16

    rnn_cell = ManualRNNCell(input_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    x_t = torch.randn(batch_size, embed_dim, device=device)
    h_0 = torch.zeros(batch_size, hidden_dim, device=device)

    h_1 = rnn_cell(x_t, h_0)
    print("\n[Manual RNN Cell Verification]")
    print(f"Input x_t shape:      {tuple(x_t.shape)}")
    print(f"Computed h_1 shape:   {tuple(h_1.shape)}")

    # 2. Test Embedding + LSTM Pipeline
    vocab_size = 50
    seq_len = 6

    model = SequenceEmbeddingRNN(
        vocab_size=vocab_size,
        embed_dim=12,
        hidden_dim=24,
        num_layers=1
    ).to(device)

    # Dummy batch of token indices: (batch_size, seq_len)
    dummy_tokens = torch.randint(low=0, high=vocab_size, size=(4, seq_len), device=device)

    lstm_out, final_repr = model(dummy_tokens)

    print("\n[Embedding + LSTM Network Verification]")
    print(f"Input Token Batch:        {tuple(dummy_tokens.shape)}")
    print(f"Full Sequence LSTM Out:   {tuple(lstm_out.shape)}  # (Batch, Seq_Len, Hidden_Dim)")
    print(f"Final Projected Vector:   {tuple(final_repr.shape)}  # (Batch, Hidden_Dim // 2)")


if __name__ == "__main__":
    main()
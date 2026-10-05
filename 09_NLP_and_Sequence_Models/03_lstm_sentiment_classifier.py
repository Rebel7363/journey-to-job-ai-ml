import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# 1. Dataset & Vocabulary Handling
RAW_DATA = [
    ("This model performs exceptionally well and converges fast", 1),
    ("Deep learning with PyTorch is clean intuitive and powerful", 1),
    ("The architecture produces state of the art accuracy results", 1),
    ("Loved the training stability and performance gains", 1),
    ("The gradient exploded and loss diverged immediately", 0),
    ("Terrible latency slow inference speed and poor accuracy", 0),
    ("The pipeline crashed with out of memory error", 0),
    ("Severely overfitted on training data with useless validation metrics", 0),
]


def build_simple_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text, _ in data:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class SentimentDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = []
        for text, label in data:
            token_ids = [vocab.get(w, 1) for w in text.lower().split()]
            self.data.append((torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Dynamic batch padding to max length in the batch
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_seqs, labels


# 2. Bi-LSTM Architecture
class BiLSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        # Bidirectional produces 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        lstm_out, (h_n, _) = self.lstm(embedded)
        
        # Concat the final forward and backward hidden states
        # h_n shape: (num_layers * 2, Batch, Hidden_Dim)
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        context = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        out = self.fc(self.dropout(context))
        return out


# 3. Training & Inference Routine
def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"--- Training Bi-LSTM Sentiment Pipeline on {device} ---\n")

    vocab = build_simple_vocab(RAW_DATA)
    dataset = SentimentDataset(RAW_DATA, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=16,
        hidden_dim=32,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train loop
    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seqs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if epoch % 5 == 0 or epoch == epochs:
            acc = (correct / total) * 100
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss / total:.4f} | Accuracy: {acc:5.1f}%")

    # Inference verification
    model.eval()
    test_phrases = [
        "PyTorch model is powerful and fast",
        "Terrible memory leak error"
    ]
    print("\n--- Live Inference Verification ---")
    with torch.no_grad():
        for phrase in test_phrases:
            ids = [vocab.get(w, 1) for w in phrase.lower().split()]
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_tensor)
            pred = torch.argmax(out, dim=1).item()
            sentiment = "Positive (1)" if pred == 1 else "Negative (0)"
            print(f"Text: '{phrase}' -> Predicted: {sentiment}")


if __name__ == "__main__":
    main()
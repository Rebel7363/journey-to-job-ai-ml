from collections import Counter
import re
from typing import Dict, List, Tuple


class TextVocabulary:
    """Builds token-to-index mapping with reserved special tokens."""

    PAD_TOKEN = "<PAD>"  # Padding token (index 0)
    UNK_TOKEN = "<UNK>"  # Unknown / Out-of-vocabulary token (index 1)
    SOS_TOKEN = "<SOS>"  # Start of sequence (index 2)
    EOS_TOKEN = "<EOS>"  # End of sequence (index 3)

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.idx2token: Dict[int, str] = {
            idx: tok for tok, idx in self.token2idx.items()
        }

    def tokenize(self, text: str) -> List[str]:
        # Lowercase and split on words/punctuation
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, corpus: List[str]) -> None:
        counter = Counter()
        for doc in corpus:
            tokens = self.tokenize(doc)
            counter.update(tokens)

        current_idx = len(self.token2idx)
        for token, count in counter.most_common():
            if count >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = current_idx
                self.idx2token[current_idx] = token
                current_idx += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        unk_idx = self.token2idx[self.UNK_TOKEN]
        indices = [self.token2idx.get(tok, unk_idx) for tok in tokens]

        if add_special_tokens:
            indices = [
                self.token2idx[self.SOS_TOKEN]
            ] + indices + [self.token2idx[self.EOS_TOKEN]]

        return indices

    def decode(self, indices: List[int]) -> str:
        tokens = [
            self.idx2token.get(idx, self.UNK_TOKEN)
            for idx in indices
            if idx != self.token2idx[self.PAD_TOKEN]
        ]
        return " ".join(tokens)


def pad_sequences(
    batch_indices: List[List[int]], pad_idx: int = 0
) -> Tuple[List[List[int]], List[int]]:
    lengths = [len(seq) for seq in batch_indices]
    max_len = max(lengths)

    padded_batch = [
        seq + [pad_idx] * (max_len - len(seq)) for seq in batch_indices
    ]
    return padded_batch, lengths


def main():
    # Sample training corpus
    corpus = [
        "PyTorch makes deep learning modular and intuitive.",
        "Natural language processing processes human speech and text.",
        "Tokenization maps raw strings to numerical sequences.",
        "Deep learning architectures process batches of padded tensors.",
    ]

    vocab = TextVocabulary(min_freq=1)
    vocab.build_vocab(corpus)

    print(f"Vocabulary Size: {len(vocab.token2idx)} tokens")
    print(f"Special Tokens: {list(vocab.token2idx.keys())[:4]}")

    # Encoding new sample
    test_sentence = "PyTorch processes text sequences intuitively."
    encoded = vocab.encode(test_sentence, add_special_tokens=True)
    decoded = vocab.decode(encoded)

    print("\n--- Tokenization & Numericalization Test ---")
    print(f"Input Text:    '{test_sentence}'")
    print(f"Token Indices: {encoded}")
    print(f"Decoded Text:  '{decoded}'")

    # Batch padding simulation
    batch = [
        vocab.encode("PyTorch is powerful.", add_special_tokens=True),
        vocab.encode(
            "Natural language processing.", add_special_tokens=True
        ),
        vocab.encode("Deep learning.", add_special_tokens=True),
    ]
    padded_batch, lengths = pad_sequences(batch, pad_idx=vocab.token2idx[vocab.PAD_TOKEN])

    print("\n--- Batch Padding Demonstration ---")
    for i, (seq, l) in enumerate(zip(padded_batch, lengths)):
        print(f"Sample {i+1} (Length {l}): {seq}")


if __name__ == "__main__":
    main()
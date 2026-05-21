from collections import Counter

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(tokenized_texts: list[list[str]], min_freq: int = 2, max_size: int = 30000) -> dict[str, int]:
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, count in counter.most_common(max_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def tokens_to_indices(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab.get(t, UNK_IDX) for t in tokens]


class BookDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[str]], labels: list[int],
                 weights: list[float], vocab: dict[str, int], max_len: int = 256):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for tokens, label, weight in zip(tokenized_texts, labels, weights):
            indices = tokens_to_indices(tokens, vocab)[:max_len]
            self.data.append((indices, label, weight))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, label, weight = self.data[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.tensor(weight, dtype=torch.float)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices_list, labels, weights = zip(*batch)
    lengths = torch.tensor([len(x) for x in indices_list], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(indices_list, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    weights = torch.stack(weights)
    return padded, labels, weights, lengths

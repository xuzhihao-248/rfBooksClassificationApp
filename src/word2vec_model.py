import os

import numpy as np
from gensim.models import Word2Vec

VECTOR_SIZE = 128
WINDOW = 5
MIN_COUNT = 2


def train_word2vec(tokenized_texts: list[list[str]], vector_size: int = VECTOR_SIZE,
                   window: int = WINDOW, min_count: int = MIN_COUNT) -> Word2Vec:
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count() or 4,
        seed=42,
    )
    return model


def texts_to_vectors(tokenized_texts: list[list[str]], w2v: Word2Vec,
                     vector_size: int = VECTOR_SIZE) -> np.ndarray:
    vectors = np.zeros((len(tokenized_texts), vector_size), dtype=np.float32)
    for i, tokens in enumerate(tokenized_texts):
        word_vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if word_vecs:
            vectors[i] = np.mean(word_vecs, axis=0)
    return vectors


def save_word2vec(w2v: Word2Vec, path: str) -> None:
    w2v.save(path)


def load_word2vec(path: str) -> Word2Vec:
    return Word2Vec.load(path)

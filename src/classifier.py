import os

import numpy as np
import torch
from gensim.models import Word2Vec

from .preprocess import tokenize, build_text
from .rf_model import predict as rf_predict, load_rf
from .trainer import load_transformer, predict_transformer
from .word2vec_model import texts_to_vectors
from .dataset import tokens_to_indices


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class BookClassifier:
    def __init__(self, model_type: str = "rf", device: torch.device | None = None):
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rf = None
        self._w2v = None
        self._transformer = None
        self._vocab = None

        if model_type == "rf":
            self._rf = load_rf(os.path.join(MODELS_DIR, "rf_model.joblib"))
            self._w2v = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
        elif model_type == "transformer":
            self._transformer, self._vocab = load_transformer(
                os.path.join(MODELS_DIR, "transformer.pt"), self.device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, title: str, description: str = "") -> dict:
        title = title.strip()
        if not title:
            title = "未知"
        row = type("Row", (), {"书名": title, "简介": description})()
        text = build_text(row)
        tokens = tokenize(text)

        if self.model_type == "rf":
            vec = texts_to_vectors([tokens], self._w2v)
            result = rf_predict(self._rf, vec)[0]
        else:
            indices = tokens_to_indices(tokens, self._vocab)
            result = predict_transformer(self._transformer, [indices], self._vocab, self.device)[0]

        return result

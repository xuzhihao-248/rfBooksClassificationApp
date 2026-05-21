import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.preprocess import load_data, tokenize, split_data
from src.word2vec_model import train_word2vec, texts_to_vectors, save_word2vec
from src.rf_model import train_random_forest, save_rf
from src.dataset import build_vocab, BookDataset, collate_fn
from src.transformer_model import SimpleTransformer
from src.trainer import train_model, save_transformer, evaluate
from src.utils import NUM_CLASSES


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "books_100k.xlsx")


def build_tokenized(df: pd.DataFrame) -> list[list[str]]:
    texts = []
    for _, row in df.iterrows():
        parts = [str(row["书名"])]
        intro = str(row.get("简介", ""))
        if intro and intro != "nan":
            parts.append(intro)
        texts.append(" ".join(parts))
    return [tokenize(t) for t in texts]


def train_rf_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    print("=== Training Word2Vec + RandomForest ===")
    print("Tokenizing...")
    train_tokens = build_tokenized(train_df)
    val_tokens = build_tokenized(val_df)

    print("Training Word2Vec...")
    w2v = train_word2vec(train_tokens)
    save_word2vec(w2v, os.path.join(MODELS_DIR, "word2vec.model"))

    print("Vectorizing...")
    X_train = texts_to_vectors(train_tokens, w2v)
    X_val = texts_to_vectors(val_tokens, w2v)
    y_train = train_df["标签"].values
    y_val = val_df["标签"].values
    weights = train_df["分类置信度"].values.astype(float)

    print("Training RandomForest...")
    rf = train_random_forest(X_train, y_train, sample_weights=weights)
    save_rf(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))

    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    print(f"RF Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


def train_transformer_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame,
                               epochs: int = 30, batch_size: int = 64, lr: float = 0.001) -> None:
    print("=== Training Simple Transformer ===")
    print("Tokenizing...")
    train_tokens = build_tokenized(train_df)
    val_tokens = build_tokenized(val_df)

    print("Building vocabulary...")
    vocab = build_vocab(train_tokens)

    train_dataset = BookDataset(train_tokens, train_df["标签"].tolist(),
                                train_df["分类置信度"].tolist(), vocab)
    val_dataset = BookDataset(val_tokens, val_df["标签"].tolist(),
                              val_df["分类置信度"].tolist(), vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(vocab_size=len(vocab), num_classes=NUM_CLASSES)

    print(f"Training on {device}...")
    history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    save_transformer(model, vocab, os.path.join(MODELS_DIR, "transformer.pt"))

    _, final_val_acc = evaluate(model, val_loader, device)
    print(f"Transformer Best Val Acc: {history['val_acc'][-1]:.4f}, Final Val Acc: {final_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train book classification models")
    parser.add_argument("--model", choices=["rf", "transformer", "all"], default="all",
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading data from {DATA_FILE}...")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} samples after confidence filter")

    train_df, val_df, test_df = split_data(df)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    if args.model in ("rf", "all"):
        train_rf_pipeline(train_df, val_df)

    if args.model in ("transformer", "all"):
        train_transformer_pipeline(train_df, val_df, epochs=args.epochs,
                                   batch_size=args.batch_size, lr=args.lr)

    print("Training complete.")


if __name__ == "__main__":
    main()

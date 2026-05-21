import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import label_to_class, class_name


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, w, lengths in dataloader:
        x, y, w, lengths = x.to(device), y.to(device), w.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss_per_sample = nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (loss_per_sample * w).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, w, lengths in dataloader:
        x, y, w, lengths = x.to(device), y.to(device), w.to(device), lengths.to(device)
        logits = model(x, lengths)
        loss_per_sample = nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (loss_per_sample * w).mean()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 30, lr: float = 0.001, device: torch.device | None = None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc

    return history


def save_transformer(model: nn.Module, vocab: dict[str, int], path: str) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "d_model": model.classifier.in_features,
        "num_classes": model.classifier.out_features,
    }, path)


def load_transformer(path: str, device: torch.device | None = None) -> tuple[nn.Module, dict[str, int]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    from .transformer_model import SimpleTransformer
    model = SimpleTransformer(
        vocab_size=len(checkpoint["vocab"]),
        num_classes=checkpoint["num_classes"],
        d_model=checkpoint["d_model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, checkpoint["vocab"]


@torch.no_grad()
def predict_transformer(model: nn.Module, indices_list: list[list[int]],
                        vocab: dict[str, int], device: torch.device) -> list[dict]:
    model.eval()
    model = model.to(device)
    results = []
    for indices in indices_list:
        if not indices:
            results.append({
                "class_code": "Z",
                "class_name": class_name("Z"),
                "confidence": 0.0,
            })
            continue
        x = torch.tensor([indices], dtype=torch.long, device=device)
        lengths = torch.tensor([len(indices)], dtype=torch.long, device=device)
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=-1)
        label = int(probs.argmax(dim=-1).item())
        code = label_to_class(label)
        results.append({
            "class_code": code,
            "class_name": class_name(code),
            "confidence": round(float(probs[0, label].item()), 4),
        })
    return results

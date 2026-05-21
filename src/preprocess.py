import re

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import extract_major_class, class_to_label


def load_data(filepath: str, min_confidence: float = 0.5) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    required = {"书名", "内容简介", "中图法分类号", "分类置信度"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["分类置信度"] >= min_confidence].copy()
    df["简介"] = df["内容简介"].fillna("").astype(str)
    df["书名"] = df["书名"].fillna("").astype(str)

    df["大类"] = df["中图法分类号"].apply(extract_major_class)
    df = df.dropna(subset=["大类"])
    df["标签"] = df["大类"].apply(class_to_label)

    return df


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^一-鿿\w]", " ", str(text))
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip()]


def build_text(row: pd.Series) -> str:
    parts = [str(row["书名"])]
    intro = str(row.get("简介", row.get("内容简介", "")))
    if intro and intro != "nan":
        parts.append(intro)
    return " ".join(parts)


def split_data(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1,
               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df, test_size=val_size + test_size, random_state=random_state,
        stratify=df["标签"]
    )
    relative_test = test_size / (val_size + test_size)
    val, test = train_test_split(
        temp, test_size=relative_test, random_state=random_state,
        stratify=temp["标签"]
    )
    return train, val, test

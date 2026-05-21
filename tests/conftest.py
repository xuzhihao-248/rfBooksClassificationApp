import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    n = 100
    np.random.seed(42)
    major_classes = list("ABCDEFGHIJ")
    codes = [f"{c}{np.random.randint(1, 99)}" for c in major_classes for _ in range(n // len(major_classes))]
    np.random.shuffle(codes)
    codes = codes[:n]
    df = pd.DataFrame({
        "书名": [f"测试书籍{i}" for i in range(n)],
        "内容简介": [f"这是第{i}本书的简介" if i % 3 != 0 else "" for i in range(n)],
        "中图法分类号": codes,
        "分类置信度": np.random.uniform(0.5, 1.0, n),
    })
    df["简介"] = df["内容简介"].fillna("").astype(str)
    df["书名"] = df["书名"].astype(str)
    df["大类"] = df["中图法分类号"].str[0]
    df["标签"] = df["大类"].apply(lambda c: ord(c) - ord("A"))
    return df

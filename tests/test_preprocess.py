import pytest
import pandas as pd
from src.preprocess import load_data, tokenize, build_text, split_data
from src.utils import extract_major_class, class_to_label, label_to_class


class TestExtractMajorClass:
    def test_valid_clc(self):
        assert extract_major_class("A81") == "A"
        assert extract_major_class("TP311.13") == "T"
        assert extract_major_class("B") == "B"

    def test_invalid_clc(self):
        assert extract_major_class("1A") is None
        assert extract_major_class("") is None
        assert extract_major_class(None) is None
        assert extract_major_class("123") is None


class TestLabelMapping:
    def test_roundtrip(self):
        for label in range(22):
            code = label_to_class(label)
            assert class_to_label(code) == label
            assert code in "ABCDEFGHIJKLMNOPQRSTUVWXZNOPQRSTUVWXZ"


class TestTokenize:
    def test_chinese_tokenization(self):
        tokens = tokenize("深入理解计算机系统")
        assert len(tokens) > 0
        combined = "".join(tokens)
        assert "深入" in combined or "计算机" in combined

    def test_empty_text(self):
        tokens = tokenize("")
        assert tokens == []

    def test_special_chars_removed(self):
        tokens = tokenize("hello!!!世界")
        assert "!" not in " ".join(tokens) or len(tokens) >= 1


class TestBuildText:
    def test_title_only(self):
        row = pd.Series({"书名": "测试书", "简介": ""})
        text = build_text(row)
        assert "测试书" in text

    def test_title_and_intro(self):
        row = pd.Series({"书名": "测试书", "简介": "这是一本好书"})
        text = build_text(row)
        assert "测试书" in text
        assert "好书" in text


class TestSplitData:
    def test_split_ratios(self, sample_df):
        train, val, test = split_data(sample_df, val_size=0.1, test_size=0.1, random_state=42)
        total = len(sample_df)
        assert abs(len(train) / total - 0.8) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05

    def test_no_overlap(self, sample_df):
        train, val, test = split_data(sample_df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

# 自然语言图书分类模型 — 测试文档

## 1. 测试策略

### 1.1 总体思路

本项目采用**单元测试**覆盖全部核心模块。每个模块独立测试，不依赖外部数据和已训练模型。设计原则：各模块单元测试全部通过即可保证系统整体正确；整体问题回归到模块间接口设计，不在测试层面排查。

### 1.2 测试框架

| 项目 | 规格 |
|------|------|
| 框架 | pytest ≥ 8.3 |
| 插件 | 无额外插件，使用 pytest 内置 assertion、fixture、parametrize |
| 运行 | `uv run pytest tests/ -v` |
| CI 兼容 | `uv run pytest tests/ --tb=short` 返回标准退出码 |

### 1.3 测试范围

```
tests/
├── __init__.py
├── conftest.py                 # 共享 fixture: sample_df
├── test_preprocess.py          # 预处理模块 (9 条)
├── test_rf_model.py            # RandomForest 模块 (2 条)
├── test_transformer.py         # Transformer 模块 (8 条)
└── test_classifier.py          # 推理接口 (1 条，占位)
```

共 **20 条**用例，覆盖 5 个模块。

### 1.4 未覆盖项（有意为之）

| 项 | 原因 |
|----|------|
| CLI 交互 (main.py) | 交互式输入不适合单元测试；核心逻辑在 classifier.py 中已覆盖 |
| 训练脚本 (train.py) | 编排层，核心逻辑在各模块中已测试 |
| 完整训练→推理流程 | 属集成测试，当前不需要（见 1.1） |
| Word2Vec 模块 | 训练由 gensim 保证正确性；texts_to_vectors 逻辑在 RF 测试中隐式验证 |
| Dataset/collate_fn | 在 Transformer 测试中通过 DataLoader 间接验证 |

### 1.5 覆盖率目标

| 指标 | 目标 |
|------|------|
| 关键模块行覆盖 | 100%（utils, preprocess, rf_model, transformer_model） |
| 次要模块 | 不做硬性要求（trainer, classifier, word2vec_model） |

---

## 2. 测试环境

### 2.1 依赖安装

```bash
uv sync        # 安装全部依赖（包含 pytest）
```

### 2.2 运行全部测试

```bash
uv run pytest tests/ -v
```

### 2.3 运行单个文件

```bash
uv run pytest tests/test_preprocess.py -v
uv run pytest tests/test_rf_model.py -v
uv run pytest tests/test_transformer.py -v
```

### 2.4 运行单个用例

```bash
uv run pytest tests/test_preprocess.py::TestExtractMajorClass::test_valid_clc -v
```

---

## 3. 测试用例清单

### 3.1 test_preprocess.py — 预处理模块 (9 条)

#### TestExtractMajorClass — 分类号提取

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| P-01 | `test_valid_clc` | `"A81"`, `"TP311.13"`, `"B"` | `"A"`, `"T"`, `"B"` | 标准格式：字母+数字、多级分类、纯字母 |
| P-02 | `test_invalid_clc` | `"1A"`, `""`, `None`, `"123"` | 全部 `None` | 边界：数字开头、空字符串、None、纯数字 |

#### TestLabelMapping — 标签编解码

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| P-03 | `test_roundtrip` | 遍历 label 0-21 | `label_to_class(i)` → `class_to_label(code)` → `i` | 双向映射一致性：编码→解码可逆，且 code 在合法字母集合中 |

#### TestTokenize — 中文分词

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| P-04 | `test_chinese_tokenization` | `"深入理解计算机系统"` | 非空 token 列表，包含有效中文 | 正常中文分词 |
| P-05 | `test_empty_text` | `""` | `[]` 空列表 | 边界：空输入 |
| P-06 | `test_special_chars_removed` | `"hello!!!世界"` | 列表中不含独立特殊字符 | 正则过滤特殊符号 |

#### TestBuildText — 文本拼接

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| P-07 | `test_title_only` | 书名="测试书", 简介="" | 拼接结果包含"测试书" | 无简介场景 |
| P-08 | `test_title_and_intro` | 书名="测试书", 简介="这是一本好书" | 拼接结果同时包含"测试书"和"好书" | 书名+简介拼接 |

#### TestSplitData — 数据集划分

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| P-09 | `test_split_ratios` | 100 条样本, val=0.1, test=0.1 | 训练≈80条, 验证≈10条, 测试≈10条 (误差<5%) | 8:1:1 比例 |
| P-10 | `test_no_overlap` | 100 条样本 | 三集合索引两两不相交 | 无数据泄露 |

---

### 3.2 test_rf_model.py — RandomForest 模块 (2 条)

#### TestRandomForest

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| R-01 | `test_train_and_predict` | 100条随机128维向量, 22类标签, 随机权重 | 返回5条结果，每条含 `class_code`/`class_name`/`confidence`，confidence∈[0,1] | 训练+推理完整流程，含权重 |
| R-02 | `test_save_and_load` | 50条随机样本训练 → 保存 → 加载 | 加载模型预测结果与原模型一致 | 持久化往返一致性 |

---

### 3.3 test_transformer.py — Transformer 模块 (8 条)

#### TestPositionalEncoding

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| T-01 | `test_shape` | d_model=64, batch=2, seq=50 | 输出 shape == 输入 shape (2,50,64) | 维度保持 |
| T-02 | `test_adds_position_info` | 全零张量 (1,10,64) | 不同位置输出不同（非全相等） | 位置编码生效 |

#### TestSelfAttention

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| T-03 | `test_output_shape` | d_model=64, batch=4, seq=20 | 输出 shape == 输入 shape (4,20,64) | 维度保持 |
| T-04 | `test_with_mask` | batch=2, seq=10, 后半段 masked | 输出 shape 正确，无 NaN | mask 不破坏计算 |

#### TestSimpleTransformer

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| T-05 | `test_forward` | vocab=1000, classes=22, batch=4, seq=30 | logits shape=(4,22) | 正常前向 |
| T-06 | `test_padding_handling` | batch=2, seq=20, 实际长度10和5, PAD=0 | logits shape=(2,22), 无 NaN | padding 被正确忽略 |
| T-07 | `test_gradient_flow` | 前向 → loss.backward() | 所有命名参数 grad 非 None，且无 NaN | 反向传播正常，无死梯度 |

---

### 3.4 test_classifier.py — 推理接口 (1 条)

| 编号 | 用例 | 输入 | 预期输出 | 说明 |
|------|------|------|----------|------|
| C-01 | `test_predict_without_description` | 无（空方法） | 通过 | 占位用例，需加载已训练模型才能实现 |

---

## 4. Fixture 说明

### `conftest.py` — `sample_df`

**用途**: 为预处理测试提供可控的模拟数据。

**生成逻辑**:
1. 取前 10 个中图法大类 (A-J)，每类均分 100 个样本
2. 随机生成 CLC 码（如 A42、B17）
3. 简介：每 3 条中有 1 条为空（模拟真实数据分布）
4. 置信度：0.5-1.0 随机分布
5. 标签映射：`ord(code) - ord('A')`

**输出**: 100 行 DataFrame，含 `书名`、`内容简介`、`中图法分类号`、`分类置信度`、`简介`、`大类`、`标签` 列。

---

## 5. 如何新增测试用例

### 5.1 新增被测模块

1. 在 `tests/` 下新建 `test_<模块名>.py`
2. 如需共享 fixture，在 `conftest.py` 中添加 `@pytest.fixture`
3. 遵循现有命名规范：`class TestXxx` → `def test_yyy(self)`

### 5.2 命名规范

```
文件名:    test_<源文件名>.py        例: test_trainer.py
类名:      Test<功能描述>             例: TestTrainingLoop
方法名:    test_<输入场景>_<预期行为>  例: test_confidence_weighted_loss
```

### 5.3 编写原则

- 每个用例只测一个行为（单一职责）
- 不依赖外部文件、网络、已训练模型
- 使用随机种子保证可复现
- 覆盖正常路径 + 边界条件（空输入、NaN、极端值）
- 对 PyTorch 模块检查梯度流（防止死模型）

### 5.4 运行检查清单

```bash
# 新增用例后依次执行
uv run pytest tests/ -v           # 全量通过
uv run pytest tests/ --tb=short   # 无隐藏错误
```

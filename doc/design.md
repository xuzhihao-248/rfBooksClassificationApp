# 自然语言图书分类模型 — 详细设计文档

## 1. 文档说明

本文档是《中文图书数据采集系统需求文档》(proposal.md) 的配套详细设计文档，面向**开发者**，描述每个模块的职责、接口、算法流程和实现细节。开发者仅凭本文档即可完成编码实现。

---

## 2. 系统架构

### 2.1 模块划分

```
rfBooksClassificationApp/
├── main.py                      # CLI 交互入口
├── train.py                     # 模型训练脚本
├── src/
│   ├── __init__.py
│   ├── utils.py                 # 常量定义、分类号映射工具
│   ├── preprocess.py            # 数据加载、清洗、分词、数据集划分
│   ├── word2vec_model.py        # Word2Vec 训练与文本向量化
│   ├── rf_model.py              # RandomForest 训练、推理、持久化
│   ├── dataset.py               # PyTorch Dataset、词表构建、批处理
│   ├── transformer_model.py     # 简易 Transformer（手写 Self-Attention）
│   ├── trainer.py               # 训练循环、评估、Transformer 持久化
│   └── classifier.py            # 统一推理接口 BookClassifier
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── conftest.py              # 测试 fixture
│   ├── test_preprocess.py
│   ├── test_rf_model.py
│   ├── test_transformer.py
│   └── test_classifier.py
├── models/                      # 训练产物（持久化模型文件）
├── data/                        # 原始数据
├── doc/
│   ├── proposal.md
│   └── design.md
├── pyproject.toml
└── uv.lock
```

### 2.2 模块依赖关系

```
utils.py          （无依赖）
preprocess.py     → utils.py
word2vec_model.py （无依赖）
rf_model.py       → utils.py
dataset.py         （无依赖）
transformer_model.py （无依赖）
trainer.py        → utils.py, transformer_model.py
classifier.py     → preprocess.py, rf_model.py, trainer.py, word2vec_model.py, dataset.py
main.py           → classifier.py
train.py          → preprocess.py, word2vec_model.py, rf_model.py, dataset.py, transformer_model.py, trainer.py, utils.py
```

---

## 3. 数据流转

### 3.1 训练流程 (train.py)

```
data/books_100k.xlsx
        │
        ▼
load_data() ───────────────────── 加载 Excel，过滤置信度 < 0.5
        │                         提取大类标签 (A-Z)，映射为数字标签 (0-21)
        ▼
split_data() ──────────────────── 分层抽样 8:1:1 → train_df, val_df, test_df
        │
        ├─── RF 流水线 ─────────────────────────────────
        │    build_tokenized() → jieba 分词
        │    train_word2vec()  → 训练 Word2Vec (gensim)
        │    texts_to_vectors() → 均值池化 → 固定维度向量
        │    train_random_forest() → sklearn RF + sample_weight
        │    保存: word2vec.model, rf_model.joblib
        │
        └─── Transformer 流水线 ────────────────────────
             build_tokenized() → jieba 分词
             build_vocab()     → 构建词表 (min_freq=2, max_size=30000)
             BookDataset()     → 文本转索引序列, 填充
             train_model()     → 训练循环, 置信度加权损失
             保存: transformer.pt (包含 model_state_dict + vocab)
```

### 3.2 推理流程 (main.py → classifier.py)

```
用户输入: 书名 (必填) + 内容简介 (选填)
        │
        ▼
build_text() ──────────────────── 拼接书名 + 简介
        │
        ▼
tokenize() ────────────────────── jieba 分词，去除特殊字符
        │
        ├─── RF 路径 ────────────────────────────────
        │    texts_to_vectors() → Word2Vec 均值池化 → 向量
        │    rf_predict()       → RandomForest 推理
        │    输出: {class_code, class_name, confidence}
        │
        └─── Transformer 路径 ────────────────────────
             tokens_to_indices() → 词表映射 → 索引序列
             predict_transformer() → 模型前向传播
             输出: {class_code, class_name, confidence}
```

---

## 4. 详细模块设计

### 4.1 utils.py — 常量与工具函数

#### 4.1.1 中图法大类映射表 `CLC_MAJOR_CLASSES`

```python
CLC_MAJOR_CLASSES: dict[str, str] = {
    "A": "马列主义、毛泽东思想",
    "B": "哲学、宗教",
    "C": "社会科学总论",
    "D": "政治、法律",
    "E": "军事",
    "F": "经济",
    "G": "文化、科学、教育、体育",
    "H": "语言、文字",
    "I": "文学",
    "J": "艺术",
    "K": "历史、地理",
    "N": "自然科学总论",
    "O": "数理科学和化学",
    "P": "天文学、地球科学",
    "Q": "生物科学",
    "R": "医药、卫生",
    "S": "农业科学",
    "T": "工业技术",
    "U": "交通运输",
    "V": "航空、航天",
    "X": "环境科学",
    "Z": "综合性图书",
}
```

共 22 个类别。注意：中图法无 `L`、`M`、`W`、`Y` 大类。

#### 4.1.2 标签编解码

```
NUM_CLASSES = 22

LABEL_TO_CLASS = ["A","B","C","D","E","F","G","H","I","J","K","N","O","P","Q","R","S","T","U","V","X","Z"]
                 ↑ 排序后的类别字母列表，索引即数字标签

CLASS_TO_LABEL = {"A":0, "B":1, ... "Z":21}
                 ↑ 反向映射
```

#### 4.1.3 函数接口

```python
def extract_major_class(clc_code: str | None) -> str | None:
    """
    从完整中图法分类号提取一级大类。
    例: "A81" → "A", "TP311.13" → "T", "1A" → None
    规则: 取首字符大写，验证是否在 CLC_MAJOR_CLASSES 中。
    """

def class_name(code: str) -> str:
    """返回大类中文名称。例: "B" → "哲学、宗教" """

def class_to_label(code: str) -> int:
    """大类字母 → 数字标签。例: "B" → 1"""

def label_to_class(label: int) -> str:
    """数字标签 → 大类字母。例: 1 → "B" """
```

---

### 4.2 preprocess.py — 数据预处理

#### 4.2.1 `load_data(filepath, min_confidence=0.5) → pd.DataFrame`

**职责**: 加载 Excel 数据，清洗过滤，提取标签。

**处理步骤**:
1. `pd.read_excel(filepath)` 加载原始 Excel
2. 校验必须字段：`书名`、`内容简介`、`中图法分类号`、`分类置信度`
3. 过滤：`df["分类置信度"] >= min_confidence`（默认 0.5）
4. 空值处理：`内容简介` NaN → 空字符串 `""`
5. 标签提取：`extract_major_class(中图法分类号)` → `大类` 列
6. 丢弃无法提取大类的行
7. 数字编码：`class_to_label(大类)` → `标签` 列（0-21）

**输出列**: 保留原列 + `简介`（fillna 后的内容简介）、`大类`（A-Z 字母）、`标签`（0-21 数字）

**异常**: 缺少必须字段时抛出 `ValueError`

#### 4.2.2 `tokenize(text: str) → list[str]`

**职责**: 中文文本分词。

**处理步骤**:
1. 正则 `[^一-鿿\w]` 去除特殊字符（保留中文和字母数字下划线）
2. `jieba.lcut(text)` 精确模式分词
3. 去除空白 token

#### 4.2.3 `build_text(row: pd.Series) → str`

**职责**: 将书名和简介拼接为一段文本，供分词使用。

**规则**: `书名 + " " + 简介`。简介为空或 NaN 时仅返回书名。

#### 4.2.4 `split_data(df, val_size=0.1, test_size=0.1, random_state=42) → tuple[DataFrame, DataFrame, DataFrame]`

**职责**: 分层抽样划分训练集/验证集/测试集。

**步骤**:
1. 第一次 `train_test_split`: 分出 `train` 80% 和 `temp` 20%，`stratify=df["标签"]`
2. 第二次 `train_test_split`: 将 `temp` 按 1:1 分为 `val` 和 `test`
3. 返回 `(train, val, test)`

---

### 4.3 word2vec_model.py — Word2Vec 训练与向量化

#### 4.3.1 超参数

```python
VECTOR_SIZE = 128   # 词向量维度
WINDOW = 5           # 上下文窗口大小
MIN_COUNT = 2        # 最低词频阈值
```

#### 4.3.2 函数接口

```python
def train_word2vec(tokenized_texts: list[list[str]], vector_size=128, window=5,
                   min_count=2) -> Word2Vec:
    """
    使用 gensim.models.Word2Vec 训练。
    - workers = CPU 核心数
    - seed = 42 (可复现)
    - 使用 Skip-gram (sg=1, 默认) 或 CBOW (sg=0)
    注：gensim 4.x 默认使用 CBOW (sg=0)，对小型语料更友好。
    """

def texts_to_vectors(tokenized_texts: list[list[str]], w2v: Word2Vec,
                     vector_size=128) -> np.ndarray:
    """
    文本 → 固定维度向量 (均值池化)。
    返回 shape=(len(texts), vector_size) 的 float32 数组。
    对每个文本: 取所有词的词向量 → 按 axis=0 求均值。
    若某文本所有词均不在词表中 → 零向量。
    """

def save_word2vec(w2v: Word2Vec, path: str) -> None:
    """使用 gensim 内置 save 方法持久化。"""

def load_word2vec(path: str) -> Word2Vec:
    """使用 gensim 内置 load 方法加载。"""
```

---

### 4.4 rf_model.py — RandomForest 分类器

#### 4.4.1 函数接口

```python
def train_random_forest(X: np.ndarray, y: np.ndarray,
                        sample_weights: np.ndarray | None = None,
                        n_estimators=200, random_state=42) -> RandomForestClassifier:
    """
    训练 sklearn RandomForest。
    参数:
        X: shape=(n_samples, 128), float32 文本向量
        y: shape=(n_samples,), int 标签 (0-21)
        sample_weights: shape=(n_samples,), 置信度权重
        n_estimators: 树的数量
        class_weight="balanced": 自动处理类别不均衡
        n_jobs=-1: 使用所有 CPU 核心
    """

def predict(model: RandomForestClassifier, X: np.ndarray) -> list[dict]:
    """
    批量推理。
    返回:
        [{
            "class_code": "B",       # 大类字母
            "class_name": "哲学、宗教", # 中文名
            "confidence": 0.8700,     # 模型概率 (float, 4位小数)
        }, ...]
    """

def save_rf(model: RandomForestClassifier, path: str) -> None:
    """使用 joblib.dump 持久化。"""

def load_rf(path: str) -> RandomForestClassifier:
    """使用 joblib.load 加载。"""
```

#### 4.4.2 置信度加权机制

RandomForest 通过 `sample_weight` 参数实现加权。权重等于分类置信度值：
- 置信度 1.0 的样本 → 权重 1.0，完整参与训练
- 置信度 0.6 的样本 → 权重 0.6，对树分裂影响降低
- 置信度 < 0.5 → 已在 preprocess.load_data 中过滤

---

### 4.5 dataset.py — PyTorch 数据集与词表

#### 4.5.1 特殊 Token

```python
PAD_TOKEN = "<PAD>"   # idx=0
UNK_TOKEN = "<UNK>"   # idx=1
```

#### 4.5.2 `build_vocab(tokenized_texts, min_freq=2, max_size=30000) → dict[str, int]`

**步骤**:
1. `collections.Counter` 统计全量词频
2. 保留 `<PAD>`(0) 和 `<UNK>`(1) 两个特殊 token
3. 按词频降序，取前 `max_size-2` 个，剔除 `freq < min_freq` 的词
4. 返回 `{word: index}` 映射

#### 4.5.3 `tokens_to_indices(tokens: list[str], vocab: dict) → list[int]`

将分词结果转为索引序列，未登录词映射为 `UNK_IDX`(1)。

#### 4.5.4 `BookDataset(Dataset)`

继承 `torch.utils.data.Dataset`。

**属性**:
- `vocab`: 词表
- `max_len`: 最大序列长度（默认 256），超出截断
- `data`: `[(indices_list, label, weight), ...]`

**构造函数参数**: `tokenized_texts`, `labels`, `weights`, `vocab`, `max_len`

**`__getitem__(idx)`** 返回: `(indices_tensor: LongTensor, label_tensor: LongTensor, weight_tensor: FloatTensor)`

#### 4.5.5 `collate_fn(batch) → tuple`

批处理函数，传给 DataLoader。

**返回**:
- `padded`: LongTensor (batch, max_seq_len)，PAD 填充
- `labels`: LongTensor (batch,)
- `weights`: FloatTensor (batch,)
- `lengths`: LongTensor (batch,) 每个样本实际长度（不含填充）

---

### 4.6 transformer_model.py — 简易 Transformer

#### 4.6.1 总体结构

```
输入: token indices (batch, seq_len)
        │
        ▼
Embedding (vocab_size → d_model)     ← 可训练词嵌入
        │
        ▼
PositionalEncoding (正弦位置编码)    ← 不可训练
        │
        ▼
SelfAttention (手写单头)             ← 三个线性层 W_q, W_k, W_v
        │
        ▼
Dropout (p=0.2)
        │
        ▼
Masked Mean Pooling                  ← 忽略 PAD token
        │
        ▼
Linear (d_model → num_classes=22)    ← 分类头
        │
        ▼
输出: logits (batch, 22)
```

#### 4.6.2 `PositionalEncoding(nn.Module)`

基于公式 `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`，`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`。

**属性**:
- `pe`: shape=(1, max_len, d_model)，注册为 `buffer`（不参与梯度计算）

**forward**: `x + pe[:, :x.size(1), :]` — 残差加法。

#### 4.6.3 `SelfAttention(nn.Module)`

单头 Scaled Dot-Product Attention，不依赖 `nn.MultiheadAttention`。

**参数**:
- `d_model`: 输入/输出维度

**层**:
- `W_q = nn.Linear(d_model, d_model, bias=False)` — Query 投影
- `W_k = nn.Linear(d_model, d_model, bias=False)` — Key 投影
- `W_v = nn.Linear(d_model, d_model, bias=False)` — Value 投影
- `scale = sqrt(d_model)`

**forward 算法**:
```
Q = W_q(x)           # (batch, seq, d_model)
K = W_k(x)
V = W_v(x)
scores = Q @ K^T / scale        # (batch, seq, seq)
if mask is not None:
    scores[mask.unsqueeze(1)] = -inf    # 遮蔽 padding 位置
attn = softmax(scores, dim=-1)
output = attn @ V                # (batch, seq, d_model)
```

**mask 机制**: `mask` shape 为 `(batch, seq)`，True 表示 PAD 位置。在 `scores` 上扩展为 `(batch, 1, seq)`，将 PAD 位置的 attention score 设为 -inf，使 softmax 后权重为 0。

#### 4.6.4 `SimpleTransformer(nn.Module)`

**构造参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | - | 词表大小 |
| `num_classes` | 22 | 分类数 |
| `d_model` | 128 | 嵌入维度 |
| `max_len` | 512 | 最大序列长度 |
| `dropout` | 0.2 | Dropout 概率 |

**`forward(x, lengths)`**:
- `x`: LongTensor (batch, seq)
- `lengths`: LongTensor (batch,) 各样本实际长度
- 返回: FloatTensor (batch, num_classes) logits

**`_padding_mask(x, lengths)`**: 生成 `(batch, seq)` 的布尔 mask，`True` = PAD 位置。

**`_masked_mean_pool(x, mask)`**: 在非 PAD 位置求均值，除数为有效 token 数（clamp 最小为 1 防止除零）。

---

### 4.7 trainer.py — 训练器

#### 4.7.1 置信度加权损失函数

```python
loss_per_sample = F.cross_entropy(logits, y, reduction="none")  # (batch,)
loss = (loss_per_sample * w).mean()  # 加权平均
```

每个样本的交叉熵损失乘以该样本的分类置信度，再对批次求均值。

#### 4.7.2 函数接口

```python
def train_epoch(model, dataloader, optimizer, device) -> (avg_loss, accuracy):
    """单轮训练。遍历 dataloader，执行前向→加权损失→反向传播→参数更新。"""

def evaluate(model, dataloader, device) -> (avg_loss, accuracy):
    """评估。@torch.no_grad()，不计算梯度。"""

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device=None) -> dict:
    """
    完整训练流程。
    - 优化器: Adam
    - 每轮执行 train_epoch + evaluate
    - 跟踪 best_val_acc
    - 返回 history dict: {train_loss: [...], train_acc: [...], val_loss: [...], val_acc: [...]}
    """

def save_transformer(model, vocab, path) -> None:
    """
    保存 checkpoint，包含:
    - model_state_dict
    - vocab (词表映射)
    - d_model
    - num_classes
    """

def load_transformer(path, device=None) -> (model, vocab):
    """加载 checkpoint，重建模型并加载权重。"""

def predict_transformer(model, indices_list, vocab, device) -> list[dict]:
    """
    单条/批量推理。
    返回:
        [{"class_code": "B", "class_name": "哲学、宗教", "confidence": 0.8700}, ...]
    空输入返回 Z 类 (综合性图书)，置信度 0.0。
    """
```

#### 4.7.3 设备选择

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

### 4.8 classifier.py — 统一推理接口

#### 4.8.1 `class BookClassifier`

**设计目的**: 封装模型加载和推理逻辑，命令行和未来 Web 界面共用同一接口。

**构造函数**:
```python
BookClassifier(model_type: str = "rf", device: torch.device | None = None)
```

| model_type | 加载文件 | 说明 |
|------------|----------|------|
| `"rf"` | `models/rf_model.joblib` + `models/word2vec.model` | RandomForest 路径 |
| `"transformer"` | `models/transformer.pt` | Transformer 路径 |
| 其他 | ValueError | 不支持的类型 |

**`predict(title: str, description: str = "") → dict`**:
```python
{
    "class_code": "B",          # 分类号
    "class_name": "哲学、宗教",  # 中文名称
    "confidence": 0.8700,       # 置信度
}
```

**内部流程**:
1. 检验 title 非空，为空时设为 "未知"
2. `build_text()` 拼接书名和简介
3. `tokenize()` jieba 分词
4. 按 model_type 选择路径推理
5. 返回结果 dict

---

### 4.9 main.py — CLI 入口

#### 4.9.1 命令行参数

```
--model {rf, transformer}   默认 rf
```

#### 4.9.2 交互流程

```
Loading rf model...
Book classification ready. Type 'quit' to exit.

请输入书名: <用户输入>
请输入内容简介（可选，按回车跳过）: <输入或回车>

==================================================
预测结果: B (哲学、宗教)
置信度: 0.87
==================================================
```

**退出**: 输入 `quit` 或 Ctrl+C。

**错误处理**: 模型文件不存在时提示先运行 train.py。

---

### 4.10 train.py — 训练脚本

#### 4.10.1 命令行参数

```
--model {rf, transformer, all}   默认 all
--epochs INT                     默认 30 (仅 transformer)
--batch-size INT                 默认 64 (仅 transformer)
--lr FLOAT                       默认 0.001 (仅 transformer)
```

#### 4.10.2 训练函数

```python
def train_rf_pipeline(train_df, val_df) -> None:
    """
    1. build_tokenized(): jieba 分词
    2. train_word2vec(): 训练 128 维词向量
    3. save_word2vec() → models/word2vec.model
    4. texts_to_vectors(): 均值池化
    5. train_random_forest(): 训练 RF (n_estimators=200, sample_weight=置信度)
    6. save_rf() → models/rf_model.joblib
    7. 打印 train/val accuracy
    """

def train_transformer_pipeline(train_df, val_df, epochs, batch_size, lr) -> None:
    """
    1. build_tokenized(): jieba 分词
    2. build_vocab(): 构建词表
    3. BookDataset(): 索引化
    4. DataLoader + collate_fn: 批处理
    5. SimpleTransformer(vocab_size, 22)
    6. train_model() → Adam 优化, 置信度加权损失
    7. save_transformer() → models/transformer.pt
    8. 打印 best val accuracy
    """
```

---

## 5. 模型持久化文件规格

### 5.1 `models/word2vec.model`

Gensim 原生格式。由 `Word2Vec.save()` 生成，`Word2Vec.load()` 加载。

### 5.2 `models/rf_model.joblib`

Joblib 序列化格式。由 `joblib.dump(model, path)` 生成。

### 5.3 `models/transformer.pt`

PyTorch checkpoint 格式（单文件）。内容结构：

```python
{
    "model_state_dict": OrderedDict,   # SimpleTransformer 权重
    "vocab": {"词": 索引, ...},         # 词表 (用于推理时 text→indices)
    "d_model": 128,                    # 嵌入维度 (用于重建模型)
    "num_classes": 22,                 # 分类数 (用于重建模型)
}
```

加载时使用 `torch.load(..., weights_only=False)` 并传入 `map_location` 确保跨设备兼容。

---

## 6. 错误处理约定

| 场景 | 处理方式 |
|------|----------|
| 模型文件缺失 | 打印提示，要求先运行 train.py |
| 书名输入为空 | 提示"书名不能为空"，重新输入 |
| 用户输入 "quit" | 退出程序 |
| 空文本分词结果 | 返回空列表 |
| 词汇全不在词表中 | 均值池化返回零向量；Transformer 返回 Z 类 |
| 中图法分类号无法解析 | 丢弃该样本（不参与训练） |
| 类别样本数不足 | 分层抽样可能失败（小数据集时手动检查） |
| Ctrl+C | 捕获 KeyboardInterrupt，优雅退出 |

---

## 7. 扩展点

### 7.1 后续 Web 界面接入

`BookClassifier` 类设计为独立可复用的推理接口：

```python
# 后续 Web 框架中直接使用
from src.classifier import BookClassifier
classifier = BookClassifier(model_type="transformer")
result = classifier.predict(title="红楼梦", description="...")
```

无需修改任何现有代码即可集成。

### 7.2 新增模型类型

1. 新建模型模块（仿照 `rf_model.py` 或 `transformer_model.py` 的模式）
2. 在 `classifier.py` 的 `BookClassifier.__init__` 中增加 `elif` 分支
3. 在 `train.py` 中增加对应的训练函数
4. 在 `main.py` 的 `--model` choices 中添加新选项

### 7.3 数据量增长

数据从当前 1.5 万增加到 10 万后：
- RF: 推理速度基本不受影响（树结构已固定）
- Transformer: 可适当增加 `epochs`、降低 `lr`、增大 `batch_size`
- Word2Vec: 不用修改，语料越大词向量质量越高

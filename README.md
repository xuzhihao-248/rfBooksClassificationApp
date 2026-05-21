# 自然语言图书分类模型

基于自然语言的图书分类模型。输入书名和内容简介（可选），输出中图法一级大类分类号（A-Z 共 22 类）。

## 方案

提供两条独立流水线，通过命令行参数切换：

| 方案 | 技术栈 | 特点 |
|------|--------|------|
| Word2Vec + RandomForest | gensim + sklearn | 训练快，可解释性好 |
| Simple Transformer | PyTorch（手写 Self-Attention） | 深度方案，数据越多越优 |

## 项目结构

```
rfBooksClassificationApp/
├── main.py                 # CLI 推理入口
├── train.py                # 训练脚本
├── src/
│   ├── preprocess.py       # 数据加载、清洗、jieba分词、数据集划分
│   ├── utils.py            # 中图法22大类映射、标签编解码
│   ├── word2vec_model.py   # Word2Vec 训练与文本向量化
│   ├── rf_model.py         # RandomForest 训练与推理
│   ├── dataset.py          # PyTorch Dataset 与词表构建
│   ├── transformer_model.py# 简易 Transformer（手写 Self-Attention）
│   ├── trainer.py          # 训练循环、评估、模型持久化
│   └── classifier.py       # 统一推理接口 BookClassifier
├── tests/                  # 单元测试（pytest）
├── models/                 # 训练产物（不纳入版本控制）
├── data/                   # 训练数据
└── doc/
    ├── proposal.md         # 需求文档
    ├── design.md           # 详细设计文档
    └── test.md             # 测试文档
```

## 快速开始

### 1. 环境要求

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)（Python 包管理器）

### 2. 安装依赖

```bash
uv sync
```

### 3. 训练模型

```bash
# 训练全部模型（RF + Transformer）
uv run python train.py --model all

# 仅训练 RandomForest
uv run python train.py --model rf

# 仅训练 Transformer（可调整超参数）
uv run python train.py --model transformer --epochs 50 --batch-size 128 --lr 0.0005
```

训练完成后，模型文件保存在 `models/` 目录：
- `models/word2vec.model` — Word2Vec 词向量
- `models/rf_model.joblib` — RandomForest 分类器
- `models/transformer.pt` — Transformer 模型

### 4. 命令行推理

```bash
# 使用 RandomForest 推理
uv run python main.py --model rf

# 使用 Transformer 推理
uv run python main.py --model transformer
```

交互示例：

```
请输入书名: 红楼梦
请输入内容简介（可选，按回车跳过）:

==================================================
预测结果: I (文学)
置信度: 0.92
==================================================
```

### 5. 运行测试

```bash
uv run pytest tests/ -v
```

## 数据说明

训练数据来源：豆瓣读书，包含约 15,500 条中文图书数据（持续扩充至 10 万条）。

### 字段

| 字段 | 用途 |
|------|------|
| 书名 | 特征（必选） |
| 内容简介 | 特征（可选） |
| 中图法分类号 | 标签（取首字母作为大类） |
| 分类置信度 | 样本权重（训练时加权损失函数） |

### 分类目标

中图法 **22 个一级大类**（A-Z，无 L/M/W/Y）：

A马列主义 · B哲学宗教 · C社科总论 · D政治法律 · E军事 · F经济 · G文化教育 · H语言文字 · I文学 · J艺术 · K历史地理 · N自然科学 · O数理化学 · P天文地理 · Q生物 · R医药卫生 · S农业 · T工业技术 · U交通运输 · V航空航天 · X环境科学 · Z综合

## 设计要点

### 置信度加权损失

训练时使用爬虫标记的分类置信度作为样本权重：置信度 ≥ 0.5 参与训练，损失函数按置信度缩放；置信度 < 0.5 的样本过滤。

### 简易 Transformer

手写 Self-Attention 机制（不依赖 `nn.TransformerEncoder`），结构：

```
Embedding → 位置编码 → Self-Attention → 均值池化 → Linear → 22分类
```

### 预留 API 接口

`src/classifier.py` 中的 `BookClassifier` 类支持直接在其他项目中导入使用：

```python
from src.classifier import BookClassifier

classifier = BookClassifier(model_type="transformer")
result = classifier.predict(title="三体", description="科幻小说")
```

## 许可证

MIT

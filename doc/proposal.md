# 自然语言图书分类模型 — 需求文档

## 1. 项目概述

### 1.1 项目目标

构建一个基于自然语言的书本分类模型。用户输入**书名**（必填）和**内容简介**（选填），模型输出对应的**中图法大类分类号**（A-Z 共 22 类）。

### 1.2 项目背景

数据采集项目（参见 `D:\code\booksClassification\doc\proposal.md`）已从豆瓣读书爬取了图书数据，目标 10 万条，覆盖中图法全部 22 个一级大类。当前数据约 1.5 万条，仍在持续爬取中。本项目基于该数据集开发分类模型。

### 1.3 核心约束

| 约束项 | 说明 |
|--------|------|
| 输入字段 | 书名（必填）+ 内容简介（选填） |
| 输出粒度 | 中图法一级大类（A-Z，22 类） |
| 实现方案 | 两条独立流水线，通过命令行参数切换 |
| 交互方式 | 命令行交互，预留 API 接口供后续 Web/HTML 前端调用 |
| 依赖管理 | uv（pyproject.toml） |
| 测试框架 | pytest |

---

## 2. 数据说明

### 2.1 数据文件

`data/books_100k.xlsx`，当前约 15,500 条，后续扩充至 10 万条。

### 2.2 字段使用

| 原始字段 | 用途 | 备注 |
|----------|------|------|
| 书名 | 模型特征（必选） | 文本向量化输入 |
| 内容简介 | 模型特征（可选） | 有则拼接，无则仅用书名 |
| 中图法分类号 | 标签 | 提取首字母作为大类标签 |
| 分类置信度 | 样本权重 | 训练时作为损失权重，< 0.5 的样本过滤 |

其他字段（作者、出版社、豆瓣标签等）在模型训练中不使用。

### 2.3 数据预处理

- 中文分词：使用 **jieba**
- 分类号提取：取中图法分类号首字母（A-Z）
- 简介为空：保留样本，模型应学会在无简介时预测
- 置信度过滤：置信度 < 0.5 的样本不参与训练
- 置信度加权：置信度 ≥ 0.5 的样本，训练损失乘以对应置信度

### 2.4 数据集划分

训练集 : 验证集 : 测试集 = **8 : 1 : 1**（分层抽样，保持各类别比例）

---

## 3. 技术方案

### 3.1 两条流水线

#### 方案一：Word2Vec + RandomForest（传统方案）

```
原始文本 → jieba分词 → Word2Vec向量化 → 文本向量(均值池化) → RandomForest分类 → 分类结果
```

- **Word2Vec**：使用图书数据自训练，不依赖预训练词向量
- **文本向量**：对书名+简介中所有词的词向量做均值池化得到固定维度向量
- **RandomForest**：sklearn 实现，作为基线模型

#### 方案二：简易 Transformer + PyTorch（深度学习方案）

```
原始文本 → jieba分词 → 词嵌入(Embedding) → 位置编码 → Self-Attention → 分类头 → 分类结果
```

- **Embedding**：PyTorch `nn.Embedding`，维度适中
- **位置编码**：手写正弦位置编码
- **Self-Attention**：手写实现（不依赖 `nn.TransformerEncoder`）
- **分类头**：线性层 + Softmax

### 3.2 两条流水线对比

| 维度 | Word2Vec + RF | Simple Transformer |
|------|---------------|-------------------|
| 框架 | gensim + sklearn | PyTorch |
| 训练速度 | 快 | 较慢 |
| 可解释性 | 较高 | 较低 |
| 对数据量要求 | 适中 | 较高（数据越多越优） |
| 推理方式 | 同流程 | 同流程，需加载模型权重 |

### 3.3 命令行参数切换

```bash
# 使用 RandomForest 模型推理
python main.py --model rf

# 使用 Transformer 模型推理
python main.py --model transformer
```

---

## 4. 命令行交互设计

### 4.1 交互流程

```
$ python main.py --model rf

请输入书名：<用户输入书名>
请输入内容简介（可选，按回车跳过）：<用户输入简介或直接回车>

==================================================
预测结果: B (哲学、宗教)
置信度: 0.87
==================================================
```

### 4.2 输出信息

- 预测分类号（如 "B"）
- 中文类别名（如 "哲学、宗教"）
- 模型置信度

---

## 5. API 接口设计（预留）

为后续 Web/HTML 前端预留，核心推理逻辑封装为独立类，命令行调用与 API 调用共享同一推理接口。

```python
# 预留接口形态（示意）
class BookClassifier:
    def predict(self, title: str, description: str = "") -> dict:
        """返回 {"class_code": "B", "class_name": "哲学、宗教", "confidence": 0.87}"""
        ...
```

后续开发 Web 界面时直接引入该模块即可。

---

## 6. 项目文件结构

```
rfBooksClassificationApp/
├── main.py                    # 入口：命令行交互、参数解析
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # 数据加载、清洗、jieba分词
│   ├── word2vec_model.py      # Word2Vec 训练与向量化
│   ├── rf_model.py            # RandomForest 训练与推理
│   ├── transformer_model.py   # 简易 Transformer（手写 Self-Attention）
│   ├── dataset.py             # PyTorch Dataset / DataLoader
│   ├── trainer.py             # 训练循环（含置信度加权损失）
│   ├── classifier.py          # 统一推理接口 BookClassifier
│   └── utils.py               # 分类号映射、常量等工具
├── data/
│   └── books_100k.xlsx        # 训练数据
├── models/                    # 训练好的模型文件存放
├── tests/
│   ├── test_preprocess.py
│   ├── test_rf_model.py
│   ├── test_transformer.py
│   └── test_classifier.py
├── doc/
│   └── proposal.md            # 本需求文档
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 7. 训练策略

### 7.1 置信度加权损失

训练时使用样本的分类置信度作为损失权重：

- 置信度 = 1.0 → 该样本完整参与梯度更新
- 置信度 = 0.7 → 该样本损失乘以 0.7，影响降低
- 置信度 < 0.5 → 该样本过滤，不参与训练

**RandomForest**：使用 `sample_weight` 参数实现加权  
**Transformer**：自定义交叉熵损失函数，按样本权重缩放

### 7.2 训练流程

1. 加载数据，清洗空值
2. 过滤置信度 < 0.5 的样本
3. 提取分类号首字母作为标签
4. jieba 分词
5. 划分训练/验证/测试集（分层抽样 8:1:1）
6. 分别训练两条流水线
7. 保存模型到 `models/` 目录
8. 在测试集上评估并记录指标

---

## 8. 评估指标

| 指标 | 说明 | 用途 |
|------|------|------|
| 准确率 (Accuracy) | 预测正确的样本占比 | 整体评估 |
| F1 值 (Macro-F1) | 各类别 F1 值的算术平均 | 衡量小类别表现 |
| 分类报告 | 每个类别的查准率/查全率/F1 | 诊断哪些类容易混淆 |

---

## 9. 测试

使用 **pytest** 进行单元测试：

- 数据预处理：分词正确性、分类号提取、空值处理
- 模型：输入/输出维度验证、模型可训练性（梯度检查）
- 训练器：置信度加权损失正确性、过滤逻辑
- 推理接口：输入验证、输出格式

---

## 10. 开发阶段

### 阶段一：数据准备
- 实现 `src/preprocess.py`：数据加载、清洗、分词、分割
- 实现 `src/utils.py`：分类号与类别名映射

### 阶段二：Word2Vec + RandomForest
- 实现 `src/word2vec_model.py`：训练 Word2Vec，文本向量化
- 实现 `src/rf_model.py`：RandomForest 训练，置信度加权

### 阶段三：简易 Transformer
- 实现 `src/dataset.py`：PyTorch Dataset
- 实现 `src/transformer_model.py`：手写 Self-Attention Transformer
- 实现 `src/trainer.py`：训练循环 + 置信度加权损失

### 阶段四：统一入口
- 实现 `src/classifier.py`：BookClassifier 统一推理接口
- 实现 `main.py`：命令行参数解析与交互

### 阶段五：测试
- 编写 tests/ 下的 pytest 用例
- 确保两条流水线均可通过测试

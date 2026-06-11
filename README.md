# 2026《人工智能导论》大作业 - 可解释的谣言检测

## 1. 项目目标

对社交媒体推文进行二分类：

- `0`：非谣言
- `1`：谣言

同时输出一段自然语言解释，说明判断依据。

当前采用 **深度学习分类器 + BM25 证据检索 + 学校 LLM 解释生成** 的复合架构。

## 2. 系统架构

```mermaid
flowchart TD
    inputText[输入推文] --> neuralModel[TF-IDF + MLP 分类器]
    neuralModel --> finalLabel[最终标签 0或1]
    neuralModel --> confidence[置信度]

    inputText --> bm25[BM25 证据检索]
    finalLabel --> bm25
    bm25 --> evidence[同标签相似样本]

    inputText --> llmPrompt[解释 Prompt]
    finalLabel --> llmPrompt
    confidence --> llmPrompt
    evidence --> llmPrompt
    llmPrompt --> sjtuLLM[学校 LLM API]
    sjtuLLM --> explanation[判断依据]
```

职责划分：

| 模块 | 作用 |
|------|------|
| `neural_classifier.py` | 本地神经网络分类器，负责最终 `0/1` 标签 |
| `bm25_retriever.py` | 检索相似训练样本，只作为解释证据 |
| `detection.py` | 复合模型调度层 |
| `llm_client.py` | 调用学校 LLM API，只生成解释 |
| `run.py` | 完整系统评估入口 |

结构：

- **LLM 不参与最终分类**，只解释已经固定的标签。
- **BM25 不投票、不改标签**，避免错误检索样本带偏分类结果。
- 如果训练集中存在完全相同的推文，则走精确匹配捷径。

## 3. 项目结构

```text
AI-Rumor-Detector/
├── data/
│   ├── train.csv              # 训练集
│   └── val.csv                # 验证集
├── results/
│   ├── deep_model_comparison/ # 各深度模型最新对比结果
│   └── prediction_results_*.csv # run.py 完整输出（含解释）
├── bm25_retriever.py          # BM25 证据检索
├── compare_deep_models.py     # 深度模型对比实验
├── detection.py               # 复合模型主逻辑
├── harness_base.py            # Harness 基类
├── llm_client.py              # 学校 LLM API
├── neural_classifier.py       # TF-IDF + MLP 主分类器
├── torch_text_classifiers.py  # TextCNN / BiLSTM 实验模型
├── run.py                     # 完整系统运行脚本
└── requirements.txt           # 依赖列表
```

## 4. 环境部署与安装

### 4.1 基础环境

建议使用 Python 3.10 或 3.12。

```bash
pip install -r requirements.txt
```

### 4.2 PyTorch 安装说明


```bash
pip install -r requirements.txt
```

如果默认源安装失败，可参考 [PyTorch 官网](https://pytorch.org/) 选择适合你系统的安装命令。例如 CPU 版本：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4.3 学校 LLM API

`llm_client.py` 已配置学校 API：

- 接口：`https://models.sjtu.edu.cn/api/v1`
- 模型：`deepseek-chat`

只有运行 `run.py` 时才需要有效 API Key。  
深度模型对比脚本 `compare_deep_models.py` 目前只对比评测情况，因此不需要调用API。

## 5. 如何运行

### 5.1 运行完整系统（标签 + 解释）

这是最终提交和展示用的命令：

```bash
python run.py
```

流程：

1. 读取 `./data/train.csv` 和 `./data/val.csv`
2. 用训练集更新 MLP 分类器和 BM25 索引
3. 对验证集逐条预测
4. 每条调用一次 LLM 生成解释
5. 保存到 `./results/prediction_results_时间戳.csv`

输出字段：

- `text`
- `true_label`
- `pred_label`
- `explanation`
- `original_index`

注意：

- 完整验证集约 401 条，受 API 限流影响，通常需要 40 分钟以上。
- `run.py` 中每条样本默认 `sleep(6)`，用于避免触发限速。

当前已保留的完整 `run.py` 结果：

- `results/prediction_results_20260611_191037.csv`

### 5.2 运行深度模型对比实验

对比脚本**不调用 LLM**，只评估分类性能，适合写报告中的模型对比表。

运行全部默认可比模型：

```bash
python compare_deep_models.py
```

等价于：

```bash
python compare_deep_models.py --models tfidf_mlp,tfidf_mlp_t045,textcnn,bilstm --torch-epochs 8
```

其他常用命令：

```bash
# 只比较 MLP 两个阈值版本
python compare_deep_models.py --models tfidf_mlp,tfidf_mlp_t045

# 只跑某一个模型
python compare_deep_models.py --models textcnn

# 快速小样本测试
python compare_deep_models.py --models tfidf_mlp --max-train-samples 200 --max-val-samples 50
```

### 5.3 对比模型说明

| 模型名 | 含义 |
|--------|------|
| `tfidf_mlp` | 当前主模型，TF-IDF + MLP，默认阈值约 `0.5` |
| `tfidf_mlp_t045` | 同一 MLP，但把谣言判定阈值降到 `0.45` |
| `textcnn` | PyTorch TextCNN |
| `bilstm` | PyTorch BiLSTM |

说明：

- `tfidf_mlp` 与 `tfidf_mlp_t045` 不是两个不同网络，而是**同一个模型、不同判决阈值**。

### 5.4 对比实验输出位置

所有深度模型最新结果统一保存在：

```text
results/deep_model_comparison/
```

主要文件：

- `summary.csv`：各模型 Accuracy / Precision / Recall / F1
- `{model}_predictions.csv`：逐样本预测与概率
- `{model}_fn.csv`：谣言漏判样本
- `{model}_fp.csv`：非谣言误报样本

当前最新对比结果：

| 模型 | Accuracy | F1 | 说明 |
|------|----------|----|------|
| `tfidf_mlp` | 0.8653 | 0.8354 | 当前主模型 |
| `tfidf_mlp_t045` | 0.8628 | 0.8338 | 阈值诊断版 |
| `bilstm` | 0.8429 | 0.8184 | 对比实验 |
| `textcnn` | 0.8354 | 0.8047 | 对比实验 |

## 6. 当前版本结果

### 6.1 完整复合系统（`run.py`）

```text
Accuracy:  0.8653
Precision: 0.8954
Recall:    0.7829
F1 Score:  0.8354
```

### 6.2 与旧版的关系

项目经历了两版演进：

| 版本 | 分类方式 | 典型 Accuracy |
|------|----------|---------------|
| 旧版 | BM25 + LLM 直接分类 | 约 0.793 |
| 当前版 | MLP 分类 + BM25 证据 + LLM 解释 | 0.8653 |

当前 Accuracy 指标主要由深度分类器决定，LLM 只负责解释，不参与改标签。



## 7. 后续调参可能方向

F1 最优阈值当前只试了 `0.5` 和 `0.45`。下一步可以在验证集上扫描找到最优阈值，然后这个对mlp两个阈值的检测就可以删了

现在表现最好的是mlp模型，精度应该还能通过调参提升，如果无法进一步提高，可能需要考虑进行更复杂的models

llm解释功能调试：优化prompt，尝试不同LLM的效果等等
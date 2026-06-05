"""
detection.py — 基于 LLM 的可解释谣言检测核心逻辑
任务：给定一条社交媒体推文，判断是谣言（1）还是非谣言（0），并输出判断依据。
策略：BM25 检索增强的 Few-shot 推理 + 精确匹配捷径 + 强制解释输出。
"""

import re
import math
import collections
import numpy as np
from harness_base import Harness


class RumorDetectionHarness(Harness):
    """
    可解释谣言检测 Harness
    
    继承自 Harness，实现 update 和 predict 方法。
    额外提供 get_last_explanation() 获取最后一次预测的解释。
    """

    def __init__(self, call_llm):
        super().__init__(call_llm)

        # BM25 索引所需数据结构
        self.doc_freqs = collections.Counter()          # 特征词 -> 包含该词的样本数
        self.postings = collections.defaultdict(list)  # 特征词 -> [(doc_idx, tf), ...]
        self.docs = []                                  # 每条样本的特征词列表
        self.doc_lens = []                              # 每条样本的特征词数量
        self.raw_texts = []                             # 原始文本
        self.labels = []                                # 标签（字符串形式，如 "0"/"1"）
        self.avgdl = 0.0
        self.N = 0

        # 精确匹配表：规范化文本 -> Counter{label: count}
        self.exact_lookup = collections.defaultdict(collections.Counter)
        # 按标签索引样本下标
        self.label_to_indices = collections.defaultdict(list)
        # 全局标签计数（用于回退）
        self.global_label_counts = collections.Counter()
        # 所有合法标签集合
        self.all_labels = set()

        # 最后预测的解释（供外部获取）
        self.last_explanation = ""

    # --------------------------------------------------------------------------
    # 特征提取（与 BM25 索引一致）
    # --------------------------------------------------------------------------
    def extract(self, text: str) -> list:
        """
        从文本中提取特征词（unigram + bigram）
        保留字母数字、中文字符、常用标点（?!），忽略其他符号。
        """
        text = (text or "").lower()
        # 匹配单词、数字、中文、感叹号、问号（社交媒体常用）
        tokens = [m.group() for m in re.finditer(r'[a-z0-9]+|[\u4e00-\u9fa5]|[?？!！]', text)]
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _norm_key(self, text: str) -> str:
        """规范化文本（只保留字母数字和中文字符），用于精确匹配查找"""
        text = (text or "").lower()
        tokens = [m.group() for m in re.finditer(r'[a-z0-9]+|[\u4e00-\u9fa5]', text)]
        return "".join(tokens)

    # --------------------------------------------------------------------------
    # 训练阶段：存储样本，构建 BM25 索引和精确匹配表
    # --------------------------------------------------------------------------
    def update(self, text: str, label: str) -> None:
        """
        接收一条带标签的训练样本，更新内部记忆
        label: 字符串形式， "0" 或 "1"
        """
        super().update(text, label)

        norm_label = label
        self.all_labels.add(norm_label)
        self.global_label_counts[norm_label] += 1

        idx = self.N
        raw_text = text or ""

        features = self.extract(raw_text)
        tf = collections.Counter(features)

        self.docs.append(features)
        self.doc_lens.append(len(features))
        self.raw_texts.append(raw_text)
        self.labels.append(norm_label)
        self.label_to_indices[norm_label].append(idx)
        self.exact_lookup[self._norm_key(raw_text)][norm_label] += 1

        self.N += 1
        self.avgdl = ((self.avgdl * (self.N - 1)) + len(features)) / self.N

        for f in set(features):
            self.doc_freqs[f] += 1

        for f, c in tf.items():
            self.postings[f].append((idx, c))

    # --------------------------------------------------------------------------
    # BM25 评分
    # --------------------------------------------------------------------------
    def _get_bm25_scores(self, query_features: list) -> np.ndarray:
        """计算查询与所有训练样本的 BM25 相关性分数"""
        k1 = 1.2
        b = 0.5

        scores = np.zeros(self.N)
        if self.N == 0 or self.avgdl <= 0:
            return scores

        q_counts = collections.Counter(query_features)

        for f, q_tf in q_counts.items():
            if f not in self.doc_freqs:
                continue
            df = self.doc_freqs[f]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            if idf < 0.01:
                idf = 0.01

            for idx, tf in self.postings.get(f, []):
                doc_len = self.doc_lens[idx]
                scores[idx] += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (doc_len / self.avgdl))
                )
        return scores

    # --------------------------------------------------------------------------
    # LLM 响应解析（提取 <reasoning> 和 <label>）
    # --------------------------------------------------------------------------
    def _parse_llm_response(self, response: str) -> tuple:
        """
        从 LLM 返回的字符串中提取推理文本和标签。
        返回 (reasoning, label)，若解析失败返回 (None, None)。
        """
        if not response:
            return None, None

        # 提取 <reasoning>...</reasoning>
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # 提取 <label>0</label> 或 <label>1</label>
        label_match = re.search(r'<label>([01])</label>', response, re.IGNORECASE)
        label = label_match.group(1) if label_match else None

        # 如果没有 <label> 标签，尝试直接寻找数字 0 或 1
        if label is None:
            digit_match = re.search(r'\b([01])\b', response)
            if digit_match:
                label = digit_match.group(1)

        return reasoning, label

    # --------------------------------------------------------------------------
    # 构造 Few-shot Prompt（强制输出推理和标签）
    # --------------------------------------------------------------------------
    def _make_fewshot_messages(self, text: str, ranked_indices: np.ndarray, scores: np.ndarray):
        """
        根据 BM25 检索结果选择代表性示例，构造 system 和 user 消息。
        返回 messages 列表，可直接传入 call_llm。
        """
        # 计算每个标签的 BM25 总分，确定 top_label（最可能的标签）
        label_score_sums = collections.defaultdict(float)
        has_positive = False
        for idx in ranked_indices:
            score = scores[int(idx)]
            if score > 0:
                has_positive = True
                label = self.labels[int(idx)]
                label_score_sums[label] += score

        sorted_valid_labels = sorted(self.all_labels, key=lambda l: label_score_sums[l], reverse=True)
        top_label = sorted_valid_labels[0] if sorted_valid_labels else None

        # 系统消息：角色与任务说明
        system_content = (
            "You are a rumor detection expert. Your task is to determine whether a given social media post "
            "is a rumor (label: 1) or non-rumor (label: 0).\n\n"
            "You must output your reasoning inside <reasoning> tags, and the final label inside <label> tags. "
            "The label should be exactly 0 or 1.\n\n"
            "Example format:\n"
            "<reasoning>The post states a confirmed fact from an official museum announcement, no signs of falsehood.</reasoning>\n"
            "<label>0</label>\n\n"
            "Now, classify the following post based on the provided reference examples."
        )

        # 收集 Few-shot 示例（控制每个标签的示例数量，避免不平衡）
        collected_examples = []
        label_quota = collections.Counter()
        used_texts = set()

        for idx in ranked_indices:
            idx = int(idx)
            if has_positive and scores[idx] <= 0:
                continue

            ex_text = self.raw_texts[idx]
            ex_label = self.labels[idx]
            ex_norm = self._norm_key(ex_text)
            if ex_norm in used_texts:
                continue

            # 配额：top_label 最多 5 个，其他标签最多 3 个
            quota_limit = 5 if ex_label == top_label else 3
            if label_quota[ex_label] >= quota_limit:
                continue

            # 构造示例文本（不包含 reasoning，因为训练样本没有解释，只提供输入和标签）
            example_str = f"Post: {ex_text}\nLabel: {ex_label}\n\n"
            collected_examples.append(example_str)
            label_quota[ex_label] += 1
            used_texts.add(ex_norm)

        # 反转使高分示例更靠近测试文本（非必需，但可读性更好）
        collected_examples.reverse()

        if not collected_examples:
            user_content = f"Post: {text}\n\nProvide your reasoning and label."
        else:
            user_content = (
                "Here are some similar posts from the training data with their labels:\n\n"
                + "".join(collected_examples)
                + f"Now determine the following post:\nPost: {text}\n\n"
                "Output your <reasoning> and <label> as shown in the example."
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # --------------------------------------------------------------------------
    # 回退方案（当 LLM 解析失败时使用）
    # --------------------------------------------------------------------------
    def _get_fallback_label_and_explanation(self, text: str, scores: np.ndarray) -> tuple:
        """返回 (回退标签, 回退解释)"""
        # 优先使用 BM25 得分最高的样本标签
        if self.N > 0:
            scores = np.nan_to_num(scores, nan=0.0)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 0:
                fallback_label = self.labels[best_idx]
                explanation = (
                    f"The most similar training sample (BM25 score {scores[best_idx]:.3f}) "
                    f"is labeled {fallback_label}. (LLM output could not be parsed.)"
                )
                return fallback_label, explanation

        # 否则使用全局出现最多的标签
        fallback_label = self.global_label_counts.most_common(1)[0][0]
        explanation = f"No similar training samples found. Falling back to majority label {fallback_label}."
        return fallback_label, explanation

    # --------------------------------------------------------------------------
    # 核心预测接口
    # --------------------------------------------------------------------------
    def predict(self, text: str) -> str:
        """
        预测输入文本的标签（0 或 1），并生成解释保存在 self.last_explanation 中。
        返回标签字符串。
        """
        self.last_explanation = ""

        if not self.all_labels:
            return ""

        # 1. 精确匹配捷径
        norm_text = self._norm_key(text)
        if norm_text in self.exact_lookup:
            cnt = self.exact_lookup[norm_text]
            exact_label = cnt.most_common(1)[0][0]
            self.last_explanation = f"Exact match found in training data, labeled as {exact_label}."
            return exact_label

        # 2. BM25 检索
        query_features = self.extract(text)
        scores = self._get_bm25_scores(query_features)
        ranked_indices = np.argsort(scores)[::-1]

        # 3. 构造 Few-shot 消息并调用 LLM
        messages = self._make_fewshot_messages(text, ranked_indices, scores)
        response = self.call_llm(messages)

        # 4. 解析 LLM 响应
        reasoning, pred_label = self._parse_llm_response(response)

        if pred_label is not None and pred_label in self.all_labels:
            # 成功解析，保存解释
            self.last_explanation = reasoning if reasoning else "No reasoning provided by LLM."
            return pred_label

        # 5. 回退方案
        fallback_label, fallback_explanation = self._get_fallback_label_and_explanation(text, scores)
        self.last_explanation = fallback_explanation
        return fallback_label

    # --------------------------------------------------------------------------
    # 获取最近一次预测的解释
    # --------------------------------------------------------------------------
    def get_last_explanation(self) -> str:
        """返回最近一次 predict 调用生成的解释文本"""
        return self.last_explanation
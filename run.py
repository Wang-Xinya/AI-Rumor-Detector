"""
run.py — 基于 LLM 的可解释谣言检测运行脚本
加载数据集，训练 RumorDetectionHarness，在验证集上评估准确率并输出示例解释。
"""

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 导入自定义模块
from llm_client import call_llm
from detection import RumorDetectionHarness


def load_data(csv_path: str):
    """加载数据集，返回 DataFrame"""
    df = pd.read_csv(csv_path)
    # 确保必要的列存在
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file {csv_path} missing required column: {col}")
    # 如果 label 是字符串，统一为小写以便处理
    df['label'] = df['label'].astype(str).str.lower()
    return df


def evaluate(harness, val_df, max_samples=None, verbose=True, save_results=True):
    """
    在验证集上评估模型，打印指标并展示部分预测示例。
    
    参数:
        harness: RumorDetectionHarness 实例
        val_df: 验证集 DataFrame
        max_samples: 最多评估多少个样本（None 表示全部）
        verbose: 是否打印详细输出
    返回:
        (accuracy, preds, trues, explanations)
    """
    if max_samples is not None and max_samples < len(val_df):
        eval_df = val_df.sample(n=max_samples, random_state=42)
    else:
        eval_df = val_df.copy()

    predictions = []
    ground_truths = []
    explanations = []
    texts = []

    print(f"\n开始评估 {len(eval_df)} 个样本...")
    for i, (idx, row) in enumerate(eval_df.iterrows()):
        # print(i)
        text = row['text']
        true_label = row['label']
        
        # 预测
        pred = harness.predict(text)
        explanations.append(harness.get_last_explanation())
        
        predictions.append(pred)
        ground_truths.append(true_label)
        texts.append(text)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  已评估 {i + 1}/{len(eval_df)} 条")
        
        # 避免调用过快导致 API 限流
        time.sleep(6)
    
    # 计算指标（将预测和真实标签转为 int 类型）
    pred_int = [int(p) for p in predictions]
    true_int = [int(t) for t in ground_truths]
    
    acc = accuracy_score(true_int, pred_int)
    prec = precision_score(true_int, pred_int, zero_division=0)
    rec = recall_score(true_int, pred_int, zero_division=0)
    f1 = f1_score(true_int, pred_int, zero_division=0)
    
    if verbose:
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\n分类报告:")
        print(classification_report(true_int, pred_int, target_names=["Non-rumor", "Rumor"]))
        
        # 展示前 5 个预测示例及其解释
        print("\n" + "=" * 60)
        print("预测示例（前5条）")
        print("=" * 60)
        for i in range(min(5, len(texts))):
            print(f"\n样本 {i+1}:")
            print(f"文本: {texts[i][:150]}..." if len(texts[i]) > 150 else f"文本: {texts[i]}")
            print(f"真实标签: {ground_truths[i]}")
            print(f"预测标签: {predictions[i]}")
            print(f"解释: {explanations[i][:200]}..." if len(explanations[i]) > 200 else f"解释: {explanations[i]}")
            print("-" * 40)

    if save_results:
        results_df = pd.DataFrame({
            'text': texts,
            'true_label': ground_truths,
            'pred_label': predictions,
            'explanation': explanations
        })
        # 可选：添加原始验证集的索引（如果 eval_df 是原 df 的子集）
        results_df['original_index'] = eval_df.index[:len(texts)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"./results/prediction_results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        if verbose:
            print(f"\n预测结果已保存至: {output_file}")
    
    return acc, predictions, ground_truths, explanations


def main():
    # 配置文件路径
    TRAIN_CSV = "./data/train.csv"
    VAL_CSV = "./data/val.csv"
    MAX_EVAL_SAMPLES = None  # 设为整数可限制评估样本数，例如 50；None 表示全量
    
    print("加载数据...")
    train_df = load_data(TRAIN_CSV)
    val_df = load_data(VAL_CSV)
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    print("\n初始化 RumorDetectionHarness...")
    harness = RumorDetectionHarness(call_llm)
    
    print("\n开始训练（构建 BM25 索引和精确匹配表）...")
    for idx, row in train_df.iterrows():
        text = row['text']
        label = row['label']
        harness.update(text, label)
        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx + 1}/{len(train_df)} 条训练样本")
    
    print(f"训练完成，共记忆 {harness.N} 个样本，合法标签: {harness.all_labels}")
    
    # 评估
    evaluate(harness, val_df, max_samples=MAX_EVAL_SAMPLES, verbose=True)
    
    print("\n运行完成。")


if __name__ == "__main__":
    main()
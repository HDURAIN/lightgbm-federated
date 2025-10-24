# -*- coding: utf-8 -*-
"""
server.py —— 服务器端：挑选客户端 → 调用客户端训练 → 聚合（软/硬投票）→ 评估
控制台仅输出两项核心指标：
  - 分类准确率（Overall Accuracy）
  - 攻击检测率（Attack Detection Rate）
详细分类报告与解释保存至 evaluation_report.txt
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from client import ClientWorker  # ← 引入客户端类

# =================== 基础配置 ===================
TRAIN_DATASETS_DIR = "./data/clients_weighted"
TEST_DATASET_PATH = "./data/data_test/test.csv"

FEATURE_LIST_PATH = "feature_list.txt"
SCHEMA_PATH = "schema.json"

# 7 类标签（按你的项目约定）
TARGET_CLASSES = ["portmap", "syn", "mssql", "ldap", "netbios", "udp", "benign"]
# =================================================

TARGET_CLASSES = [c.lower() for c in TARGET_CLASSES]
TARGET_CLASS2ID = {name: i for i, name in enumerate(TARGET_CLASSES)}
TARGET_ID2CLASS = {i: name for name, i in TARGET_CLASS2ID.items()}
NUM_CLASSES = len(TARGET_CLASSES)
BENIGN_ID = TARGET_CLASS2ID["benign"]

def parse_label_series_7(series: pd.Series) -> np.ndarray:
    out = []
    for t in series:
        key = str(t).strip().split('_')[-1].replace('-', '').replace(' ', '').lower()
        out.append(TARGET_CLASS2ID.get(key, -1))
    return np.array(out, dtype=np.int32)

def numeric_cols(df: pd.DataFrame):
    cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    return [c for c in cols if c != "Label"]

def build_schema_if_needed():
    if os.path.exists(FEATURE_LIST_PATH) and os.path.exists(SCHEMA_PATH):
        return

    client_files = [f for f in os.listdir(TRAIN_DATASETS_DIR)
                    if f.lower().endswith(".csv") and "ds_store" not in f.lower()]
    client_files.sort()
    feature_union = None
    for f in client_files:
        df_head = pd.read_csv(os.path.join(TRAIN_DATASETS_DIR, f), nrows=1, low_memory=False)
        df_head.columns = [c.strip() for c in df_head.columns]
        cols = set(numeric_cols(df_head))
        feature_union = cols if feature_union is None else (feature_union | cols)

    test_head = pd.read_csv(TEST_DATASET_PATH, nrows=1, low_memory=False)
    test_head.columns = [c.strip() for c in test_head.columns]
    test_cols = set(numeric_cols(test_head))
    feature_list = sorted(list((feature_union or set()) & test_cols))
    assert feature_list, "客户端与测试集数值特征交集为空，请检查列名/空格/类型。"

    schema = {c: {"dtype": "float64", "fill": 0.0} for c in feature_list}
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        for c in feature_list:
            f.write(c + "\n")
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

def load_feature_spec():
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_list = [line.strip() for line in f if line.strip()]
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return feature_list, schema

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description="联邦服务器（单轮、模型级聚合）")
    parser.add_argument("--clients", type=int, default=None,
                        help="参与的客户端数量（上限600；未指定则取 min(600, 可用文件数)）")
    parser.add_argument("--vote-mode", type=str, default="soft_weighted",
                        choices=["soft_weighted", "soft_logit", "hard"],
                        help="投票方式：soft_weighted(默认)，soft_logit，hard(硬投票)")
    parser.add_argument("--keep-top-pct", type=float, default=0.8,
                        help="按验证集acc筛端比例 0~1；小客户端实验建议设 1.0")
    parser.add_argument("--no-filter", action="store_true",
                        help="禁用筛端（覆盖 keep-top-pct，聚合全部参与客户端）")
    args = parser.parse_args()

    VOTE_MODE = args.vote_mode
    KEEP_TOP_PCT = max(0.0, min(1.0, args.keep_top_pct))
    ENABLE_FILTER = (not args.no_filter)

    # 准备特征规范
    build_schema_if_needed()
    FEATURE_LIST, SCHEMA = load_feature_spec()

    # 可用客户端文件列表
    all_client_files = [f for f in os.listdir(TRAIN_DATASETS_DIR)
                        if f.lower().endswith(".csv") and "ds_store" not in f.lower()]
    all_client_files.sort()
    total_available = len(all_client_files)
    if total_available == 0:
        raise RuntimeError(f"在 {TRAIN_DATASETS_DIR} 未找到任何客户端CSV。")

    cap = min(600, total_available)
    num_user = cap if (args.clients is None) else max(1, min(int(args.clients), cap))

    print(f"==> 启动联邦学习模拟（LightGBM），参与客户端：{num_user}/{total_available}")

    # ---------- 客户端训练 ----------
    workers = []
    val_accs = np.zeros(num_user, dtype=np.float64)
    train_bar = tqdm(total=num_user, desc="客户端训练进度", ncols=80, unit="终端")
    for idx in range(num_user):
        w = ClientWorker(
            user_idx=idx,
            feature_list=FEATURE_LIST,
            schema=SCHEMA,
            class_weight=None,                # 如需类权重，可在此处传入预先计算的字典
            target_class2id=TARGET_CLASS2ID,
        )
        acc = w.train_and_eval()
        workers.append(w)
        val_accs[idx] = acc
        train_bar.update(1)
    train_bar.close()

    # ---------- 加载测试数据 ----------
    raw_test = pd.read_csv(TEST_DATASET_PATH, low_memory=False)
    raw_test.columns = [c.strip() for c in raw_test.columns]
    y_all = parse_label_series_7(raw_test["Label"])
    mask = (y_all >= 0)
    test_df = raw_test.loc[mask].reset_index(drop=True)
    y_true = y_all[mask]
    N = len(y_true)

    # ---------- 选择参与聚合的客户端（Top PCT by val_acc） ----------
    order = np.argsort(-val_accs)  # 降序
    if ENABLE_FILTER and 0 < KEEP_TOP_PCT < 1:
        k = max(1, int(num_user * KEEP_TOP_PCT))
        client_indices = order[:k]
    else:
        client_indices = order  # 全部参与

    # ---------- 聚合 ----------
    agg_bar = tqdm(total=len(client_indices), desc="服务器聚合进度", ncols=80, unit="模型")

    if VOTE_MODE in ("soft_weighted", "soft_logit"):
        # 准备权重（均权或按 val_acc 加权）
        if VOTE_MODE == "soft_weighted":
            sel_accs = val_accs[client_indices]
            w = sel_accs / max(sel_accs.mean(), 1e-6)
            w = np.clip(w, 1e-3, None)
        else:
            w = np.ones(len(client_indices), dtype=np.float64)

        sum_logits = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        for j, u in enumerate(client_indices):
            raw = workers[int(u)].predict_raw_df(test_df)  # logits: [N, C]
            sum_logits += raw * w[j]
            agg_bar.update(1)
        agg_bar.close()

        probs = softmax(sum_logits / np.sum(w))
        y_pred = np.argmax(probs, axis=1)

    elif VOTE_MODE == "hard":
        # 硬投票：每端一票；平票使用平均logits打破
        vote_counts = np.zeros((N, NUM_CLASSES), dtype=np.int32)
        sum_logits_tie = np.zeros((N, NUM_CLASSES), dtype=np.float64)

        for u in client_indices:
            logits = workers[int(u)].predict_raw_df(test_df)
            pred_cls = np.argmax(logits, axis=1)
            vote_counts[np.arange(N), pred_cls] += 1
            sum_logits_tie += logits
            agg_bar.update(1)
        agg_bar.close()

        y_pred = np.argmax(vote_counts, axis=1)
        max_votes = vote_counts.max(axis=1, keepdims=True)
        ties = (vote_counts == max_votes)
        tie_rows = np.where(ties.sum(axis=1) > 1)[0]
        if tie_rows.size > 0:
            for r in tie_rows:
                cls_candidates = np.where(ties[r])[0]
                best_cls = cls_candidates[np.argmax(sum_logits_tie[r, cls_candidates])]
                y_pred[r] = best_cls

    else:
        raise ValueError(f"未知投票模式: {VOTE_MODE}")

    # ---------- 指标：分类准确率 + 攻击检测率 ----------
    acc = accuracy_score(y_true, y_pred)

    attack_mask = (y_true != BENIGN_ID)
    attack_total = int(np.sum(attack_mask))
    attack_detected = int(np.sum((y_pred != BENIGN_ID) & attack_mask))
    attack_detection_rate = (attack_detected / attack_total) if attack_total > 0 else 0.0

    # ---------- 保存逐样本预测 ----------
    mapped_results = [TARGET_ID2CLASS[int(c)] for c in y_pred.tolist()]
    pd.DataFrame(mapped_results, columns=["Prediction"]).to_csv("results_round1.csv", index=False)

    # ---------- 保存详细报告（含解释，静默UndefinedMetricWarning） ----------
    cls_report = classification_report(
        y_true, y_pred,
        target_names=[TARGET_ID2CLASS[i] for i in range(NUM_CLASSES)],
        digits=4,
        zero_division=0
    )
    explanation = """
评估指标解释：
1) 分类准确率（Overall Accuracy）:
   - 定义：所有样本中预测正确的比例。
   - 作用：衡量模型整体在7类上的识别准确性。
2) 攻击检测率（Attack Detection Rate）:
   - 定义：所有攻击样本中被正确识别为攻击的比例。
     Attack Detection Rate = 被正确识别为攻击的攻击样本数 / 攻击样本总数。
   - 作用：衡量系统发现攻击的能力，越高越好（1.0 表示无漏检）。
""".strip()
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Overall Summary ===\n")
        f.write(f"Vote Mode               : {VOTE_MODE}\n")
        f.write(f"Clients (used/total)    : {len(client_indices)}/{num_user}\n")
        f.write(f"Overall Accuracy        : {acc:.4f}\n")
        f.write(f"Attack Detection Rate   : {attack_detection_rate:.4f}\n\n")
        f.write("=== Per-class Report (Precision / Recall / F1 / Support) ===\n")
        f.write(cls_report + "\n\n")
        f.write("=== Explanation ===\n")
        f.write(explanation + "\n")

    # ---------- 控制台仅输出核心指标 ----------
    print(f"分类准确率（Overall Accuracy）: {acc:.4f}")
    print(f"攻击检测率（Attack Detection Rate）: {attack_detection_rate:.4f}")

if __name__ == "__main__":
    main()

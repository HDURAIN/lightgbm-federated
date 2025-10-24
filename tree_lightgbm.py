# tree_lightgbm.py
import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from worker import Worker

# =================== 基础配置 ===================
TRAIN_DATASETS_DIR = "./data/clients_weighted"
TEST_DATASET_PATH = "./data/data_test/test.csv"

VOTE_MODE = "soft_logit"      # soft / soft_weighted / soft_logit / soft_conf / soft_weighted_conf / hard
ENABLE_FILTER_CLIENTS = True
KEEP_TOP_PCT = 0.8

ENABLE_CLASS_WEIGHT = False

FEATURE_LIST_PATH = "feature_list.txt"
SCHEMA_PATH = "schema.json"

TARGET_CLASSES = ["portmap", "syn", "mssql", "ldap", "netbios", "udp", "benign"]
# ===============================================


TARGET_CLASSES = [c.lower() for c in TARGET_CLASSES]
TARGET_CLASS2ID = {name: i for i, name in enumerate(TARGET_CLASSES)}
TARGET_ID2CLASS = {i: name for name, i in TARGET_CLASS2ID.items()}
NUM_CLASSES = len(TARGET_CLASSES)


def parse_label_series_7(series: pd.Series) -> np.ndarray:
    out = []
    for t in series:
        key = str(t).split('_')[-1].replace('-', '').lower()
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
        cols = set(numeric_cols(df_head))
        feature_union = cols if feature_union is None else (feature_union | cols)

    test_head = pd.read_csv(TEST_DATASET_PATH, nrows=1, low_memory=False)
    test_cols = set(numeric_cols(test_head))
    feature_list = sorted(list((feature_union or set()) & test_cols))
    assert feature_list, "客户端与测试集特征交集为空，请检查数据列名。"

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


# ============== 主流程 ==============
parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=1, help="Federated learning rounds")
args = parser.parse_args()
FED_ROUNDS = max(1, args.rounds)

build_schema_if_needed()
FEATURE_LIST, SCHEMA = load_feature_spec()

client_files = [f for f in os.listdir(TRAIN_DATASETS_DIR)
                if f.lower().endswith(".csv") and "ds_store" not in f.lower()]
client_files.sort()
num_user = len(client_files)

# 计算全局类别权重（可选）
CLASS_WEIGHT_DICT = None
if ENABLE_CLASS_WEIGHT:
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in client_files:
        df = pd.read_csv(os.path.join(TRAIN_DATASETS_DIR, f), usecols=["Label"], low_memory=False)
        y_all = parse_label_series_7(df["Label"])
        y = y_all[y_all >= 0]
        for c in range(NUM_CLASSES):
            class_counts[c] += np.count_nonzero(y == c)
    inv = 1.0 / np.maximum(class_counts.astype(np.float64), 1e-6)
    class_weight = inv / inv.mean()
    CLASS_WEIGHT_DICT = {int(i): float(w) for i, w in enumerate(class_weight)}

print(f"==> 启动联邦学习模拟，共 {FED_ROUNDS} 轮，{num_user} 个客户端")
t0_all = datetime.datetime.now()
round_results = []

# ---------- 每轮联邦训练 ----------
for rnd in range(1, FED_ROUNDS + 1):
    print(f"\n--- 第 {rnd} 轮训练开始 ---")
    workers = [
        Worker(
            user_idx=u,
            feature_list=FEATURE_LIST,
            schema=SCHEMA,
            class_weight=CLASS_WEIGHT_DICT,
            target_class2id=TARGET_CLASS2ID,
        )
        for u in range(num_user)
    ]

    # 客户端训练
    train_bar = tqdm(total=num_user, desc="客户端训练进度", ncols=80, unit="终端")
    for u in range(num_user):
        workers[u].user_round_train_eval()
        train_bar.update(1)
    train_bar.close()

    # 加载测试数据
    raw_test = pd.read_csv(TEST_DATASET_PATH, low_memory=False)
    y_all = parse_label_series_7(raw_test["Label"])
    mask = (y_all >= 0)
    test_df = raw_test.loc[mask].reset_index(drop=True)
    y_true = y_all[mask]
    N = len(y_true)

    # 模拟云服务器汇总
    client_indices = np.arange(num_user)
    if ENABLE_FILTER_CLIENTS and 0 < KEEP_TOP_PCT < 1:
        k = max(1, int(num_user * KEEP_TOP_PCT))
        client_indices = np.sort(client_indices[-k:])

    agg_bar = tqdm(total=len(client_indices), desc="服务器聚合进度", ncols=80, unit="模型")
    num_classes = NUM_CLASSES

    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    sum_logits = np.zeros((N, num_classes), dtype=np.float64)
    for u in client_indices:
        raw = workers[u].predict_raw_df(test_df)
        sum_logits += raw
        agg_bar.update(1)
    agg_bar.close()

    probs = softmax(sum_logits / len(client_indices))
    final_pred_result = np.argmax(probs, axis=1).tolist()

    # 多类指标
    acc = accuracy_score(y_true, final_pred_result)
    macro_f1 = f1_score(y_true, final_pred_result, average="macro")

    # 恶意识别指标
    BENIGN_ID = TARGET_CLASS2ID["benign"]
    true_malicious = (y_true != BENIGN_ID).astype(int)
    pred_malicious = (np.array(final_pred_result) != BENIGN_ID).astype(int)
    malicious_acc = (true_malicious == pred_malicious).mean()

    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, final_pred_result):
        if yt == BENIGN_ID and yp == BENIGN_ID: tp += 1
        elif yt != BENIGN_ID and yp == BENIGN_ID: fp += 1
        elif yt != BENIGN_ID and yp != BENIGN_ID and yt == yp: tn += 1
        elif yt == BENIGN_ID and yp != BENIGN_ID: fn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0

    mapped_results = [TARGET_ID2CLASS[c] for c in final_pred_result]
    out_csv = f"results_round{rnd}.csv"
    pd.DataFrame(mapped_results, columns=["Prediction"]).to_csv(out_csv, index=False)

    # 简洁输出
    print(f"--- 第 {rnd} 轮结果 ---")
    print(f"分类准确率：{acc:.4f}")
    print(f"恶意识别准确率：{malicious_acc:.4f}")
    print(f"Benign 类：Recall={recall:.4f}  Precision={precision:.4f}  FPR={fpr:.4f}")
    print(f"Macro-F1：{macro_f1:.4f}")
    print(f"结果文件：{out_csv}")
    round_results.append((acc, malicious_acc, macro_f1))

# ---------- 汇总 ----------
t1_all = datetime.datetime.now()
total_s = (t1_all - t0_all).total_seconds()
acc, malicious_acc, macro_f1 = round_results[-1]

print("\n================ 联邦学习总结 ================")
print(f"轮数：{FED_ROUNDS}  |  总耗时：{total_s:.1f}s")
print(f"最终分类准确率：{acc:.4f}")
print(f"最终恶意识别准确率：{malicious_acc:.4f}")
print(f"最终 Macro-F1：{macro_f1:.4f}")
print("=============================================\n")

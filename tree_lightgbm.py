# tree_lightgbm.py
import os
import re
import json
import argparse
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from worker import Worker

# ---------------- 路径与类别 ----------------
TRAIN_DIR = "./data/clients_weighted"
TEST_PATH = "./data/data_test/test.csv"
FEATURE_LIST_PATH = "feature_list.txt"
SCHEMA_PATH = "schema.json"

TARGET_CLASSES = ["portmap", "syn", "mssql", "ldap", "netbios", "udp", "benign"]
TARGET_CLASSES = [c.lower() for c in TARGET_CLASSES]
CLS2ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
ID2CLS = {i: c for c, i in CLS2ID.items()}
NUM_CLASSES = len(TARGET_CLASSES)
# -------------------------------------------

# ============ 列名统一（关键修复） ============
def _norm_col_name(c: str):
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    if c.lower().startswith("unnamed:"):
        return None
    return c

def sanitize_and_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    orig = list(df.columns)
    normed, keep_idx = [], []
    for i, c in enumerate(orig):
        nc = _norm_col_name(c)
        if nc is None:
            continue
        normed.append(nc)
        keep_idx.append(i)
    if len(keep_idx) < len(orig):
        df = df.iloc[:, keep_idx].copy()

    # 避免规范化后重名
    seen, final = {}, []
    for c in normed:
        if c not in seen:
            seen[c] = 1
            final.append(c)
        else:
            k = seen[c]; seen[c] += 1
            final.append(f"{c}__dup{k}")
    df.columns = final
    return df

def read_csv_sanitized(path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return sanitize_and_drop_columns(df)
# ===========================================

# ---------------- 工具函数 ----------------
def parse_label_7(series: pd.Series) -> np.ndarray:
    out = []
    for t in series:
        key = str(t).split("_")[-1].replace("-", "").lower()
        out.append(CLS2ID.get(key, -1))
    return np.array(out, dtype=np.int32)

def numeric_cols(df: pd.DataFrame):
    cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    return [c for c in cols if c != "Label"]

def build_schema_if_needed():
    if os.path.exists(FEATURE_LIST_PATH) and os.path.exists(SCHEMA_PATH):
        return
    client_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".csv")]
    client_files.sort()

    feat_union = None
    for f in client_files:
        head = read_csv_sanitized(os.path.join(TRAIN_DIR, f), nrows=1)
        cols = set(numeric_cols(head))
        feat_union = cols if feat_union is None else (feat_union | cols)

    test_head = read_csv_sanitized(TEST_PATH, nrows=1)
    test_cols = set(numeric_cols(test_head))
    feature_list = sorted(list((feat_union or set()) & test_cols))
    assert feature_list, "客户端与测试集的数值特征交集为空，请检查列名。"

    schema = {c: {"dtype": "float64", "fill": 0.0} for c in feature_list}
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        for c in feature_list:
            f.write(c + "\n")
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

def load_feature_spec():
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feats = [x.strip() for x in f if x.strip()]
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return feats, schema

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def multiclass_logloss(y_true, probs, eps=1e-12):
    N = len(y_true)
    p = np.clip(probs[np.arange(N), y_true], eps, 1.0)
    return -float(np.mean(np.log(p)))

def auto_build_server_val(client_files, per_class=400, seed=42):
    rng = np.random.RandomState(seed)
    buckets = {i: [] for i in range(NUM_CLASSES)}
    for f in tqdm(client_files, desc="构建验证集采样", ncols=80, unit="文件"):
        df = read_csv_sanitized(os.path.join(TRAIN_DIR, f))
        if "Label" not in df.columns:
            continue
        y = parse_label_7(df["Label"])
        m = (y >= 0)
        if m.sum() == 0:
            continue
        tmp = df.loc[m]
        for cid in range(NUM_CLASSES):
            idx = np.where(y[m] == cid)[0]
            if len(idx) == 0:
                continue
            take = min(len(idx), per_class // max(1, len(client_files)//2) + 1)
            if take > 0:
                pick = rng.choice(idx, size=take, replace=False)
                buckets[cid].append(tmp.iloc[pick])
    parts = []
    for cid in range(NUM_CLASSES):
        if len(buckets[cid]) == 0:
            continue
        cat = pd.concat(buckets[cid], axis=0, ignore_index=True)
        parts.append(cat.sample(n=min(len(cat), per_class), random_state=seed))
    val = pd.concat(parts, axis=0, ignore_index=True)
    val = val.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val.to_csv("server_val.auto.csv", index=False)
    return val
# -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1, help="联邦轮数")
    ap.add_argument("--fusion", type=str, default="avg", choices=["avg", "weighted"])
    ap.add_argument("--alpha", type=float, default=1.0, help="weighted 融合的温度系数")
    ap.add_argument("--server-val", type=str, default=None, help="固定验证集路径；缺省自动构建")
    ap.add_argument("--val-per-class", type=int, default=400, help="自动构建验证集时每类样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--round-reseed", action="store_true", help="每轮更换随机种子（seed+round）")
    args = ap.parse_args()

    ROUNDS = max(1, args.rounds)
    FUSION = args.fusion
    ALPHA = float(args.alpha)
    SEED = int(args.seed)

    random.seed(SEED); np.random.seed(SEED)

    # 统一列名后重建/加载特征白名单
    build_schema_if_needed()
    FEATS, SCHEMA = load_feature_spec()

    client_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".csv")]
    client_files.sort()
    num_clients = len(client_files)

    print(f"==> 启动联邦学习模拟，共 {ROUNDS} 轮，{num_clients} 个客户端")
    print(f"融合模式：{FUSION} | Alpha：{ALPHA:.2f}")

    # 验证集
    if args.server_val and os.path.exists(args.server_val):
        server_val_df = read_csv_sanitized(args.server_val)
    else:
        server_val_df = auto_build_server_val(client_files, per_class=args.val_per_class, seed=SEED)

    # 测试集
    test_raw = read_csv_sanitized(TEST_PATH)
    y_all = parse_label_7(test_raw["Label"])
    m7 = (y_all >= 0)
    test_df = test_raw.loc[m7].reset_index(drop=True)
    y_true = y_all[m7]
    N_test = len(y_true)

    t0 = datetime.datetime.now()
    probs_round_sum = np.zeros((N_test, NUM_CLASSES), dtype=np.float64)

    for r in range(1, ROUNDS + 1):
        if args.round_reseed:
            round_seed = SEED + r
            random.seed(round_seed); np.random.seed(round_seed)

        print(f"\n--- 第 {r} 轮训练开始 ---")
        # 训练所有客户端
        workers = [
            Worker(u, FEATS, SCHEMA, target_class2id=CLS2ID,
                   csv_reader=read_csv_sanitized, train_dir=TRAIN_DIR)
            for u in range(num_clients)
        ]
        bar = tqdm(total=num_clients, desc="客户端训练进度", ncols=80, unit="终端")
        local_acc = []
        for u in range(num_clients):
            local_acc.append(workers[u].user_round_train_eval())
            bar.update(1)
        bar.close()

        # 准备 server_val / test logits
        yv_all = parse_label_7(server_val_df["Label"])
        mv = (yv_all >= 0)
        val_df = server_val_df.loc[mv].reset_index(drop=True)
        y_val = yv_all[mv]

        logits_val = []
        logits_test = []
        agg = tqdm(total=num_clients, desc="服务器聚合进度", ncols=80, unit="模型")
        for u in range(num_clients):
            logits_val.append(workers[u].predict_raw_df(val_df))
            logits_test.append(workers[u].predict_raw_df(test_df))
            agg.update(1)
        agg.close()

        # avg 概率
        sum_logits = np.zeros((N_test, NUM_CLASSES))
        for raw in logits_test:
            sum_logits += raw
        probs_avg = softmax(sum_logits / float(num_clients))

        if FUSION == "avg":
            probs = probs_avg

        else:  # weighted
            # 以 server_val 上的多类 logloss 学权重
            p_val_list = [softmax(z) for z in logits_val]
            losses = np.array([
                multiclass_logloss(y_val, p) for p in p_val_list
            ])
            z = -ALPHA * (losses - losses.min())
            w = np.exp(z); w /= (w.sum() + 1e-12)

            sum_logits_w = np.zeros((N_test, NUM_CLASSES))
            for wi, raw in zip(w, logits_test):
                sum_logits_w += wi * raw
            probs = softmax(sum_logits_w)

        y_pred = probs.argmax(axis=1)

        # 评估
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        BENIGN = CLS2ID["benign"]
        true_mal = (y_true != BENIGN).astype(int)
        pred_mal = (y_pred != BENIGN).astype(int)
        mal_acc = (true_mal == pred_mal).mean()
        # Benign 指标
        tp = fp = tn = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == BENIGN and yp == BENIGN: tp += 1
            elif yt != BENIGN and yp == BENIGN: fp += 1
            elif yt != BENIGN and yp != BENIGN and yt == yp: tn += 1
            elif yt == BENIGN and yp != BENIGN: fn += 1
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / (tn + fp) if (tn + fp) else 0.0

        out_csv = f"results_round{r}.csv"
        pd.DataFrame([ID2CLS[i] for i in y_pred], columns=["Prediction"]).to_csv(out_csv, index=False)

        print(f"--- 第 {r} 轮结果 ---")
        print(f"分类准确率：{acc:.4f}")
        print(f"恶意识别准确率：{mal_acc:.4f}")
        print(f"Benign 类：Recall={recall:.4f}  Precision={precision:.4f}  FPR={fpr:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"结果文件：{out_csv}")

        probs_round_sum += probs

    # 总结 & 轮间平均（如 rounds>1）
    t1 = datetime.datetime.now()
    print("\n================ 联邦学习总结 ================")
    print(f"轮数：{ROUNDS}  |  总耗时：{(t1 - t0).total_seconds():.1f}s")

    # 用最后一轮的结果作为“最终”
    y_last = (probs).argmax(axis=1)
    print(f"最终分类准确率：{accuracy_score(y_true, y_last):.4f}")
    BENIGN = CLS2ID["benign"]
    print(f"最终恶意识别准确率：{((y_true != BENIGN) == (y_last != BENIGN)).mean():.4f}")
    print(f"最终 Macro-F1：{f1_score(y_true, y_last, average='macro'):.4f}")

    if ROUNDS > 1:
        probs_ens = probs_round_sum / float(ROUNDS)
        y_ens = probs_ens.argmax(axis=1)
        print("---- 轮间集成（平均概率） ----")
        print(f"集成后分类准确率：{accuracy_score(y_true, y_ens):.4f}")
        print(f"集成后恶意识别准确率：{((y_true != BENIGN) == (y_ens != BENIGN)).mean():.4f}")
        print(f"集成后 Macro-F1：{f1_score(y_true, y_ens, average='macro'):.4f}")
    print("=============================================")

if __name__ == "__main__":
    main()

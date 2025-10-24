# build_server_val_hard.py
import os
import json
import argparse
import hashlib
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from worker import Worker

# ------------ 项目内的固定设定（与 tree_lightgbm.py 对齐） ------------
TRAIN_DIR_DEFAULT = "./data/clients_weighted"
TEST_PATH_DEFAULT = "./data/data_test/test.csv"
OUT_PATH_DEFAULT = "./data/server_val_hard.csv"
FEATURE_LIST_PATH = "feature_list.txt"
SCHEMA_PATH = "schema.json"

TARGET_CLASSES = ["portmap", "syn", "mssql", "ldap", "netbios", "udp", "benign"]
TARGET_CLASSES = [c.lower() for c in TARGET_CLASSES]
CLASS2ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}
NUM_CLASSES = len(TARGET_CLASSES)
# --------------------------------------------------------------------


def parse_label_series_7(series: pd.Series) -> np.ndarray:
    out = []
    for t in series:
        key = str(t).split("_")[-1].replace("-", "").lower()
        out.append(CLASS2ID.get(key, -1))
    return np.array(out, dtype=np.int32)


def numeric_cols(df: pd.DataFrame):
    cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    return [c for c in cols if c != "Label"]


def ensure_feature_spec(train_dir: str, test_path: str):
    if os.path.exists(FEATURE_LIST_PATH) and os.path.exists(SCHEMA_PATH):
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            feature_list = [x.strip() for x in f if x.strip()]
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return feature_list, schema

    client_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".csv")]
    client_files.sort()
    feat_union = None
    for f in client_files:
        df = pd.read_csv(os.path.join(train_dir, f), nrows=1, low_memory=False)
        cols = set(numeric_cols(df))
        feat_union = cols if feat_union is None else (feat_union | cols)
    test_head = pd.read_csv(test_path, nrows=1, low_memory=False)
    test_cols = set(numeric_cols(test_head))
    feature_list = sorted(list((feat_union or set()) & test_cols))
    assert feature_list, "客户端与测试集的数值特征交集为空，请检查数据列。"

    schema = {c: {"dtype": "float64", "fill": 0.0} for c in feature_list}
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        for c in feature_list:
            f.write(c + "\n")
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return feature_list, schema


def row_hash(df: pd.DataFrame, cols: list) -> pd.Series:
    """对数值特征行做 md5 指纹（严格相等检测）"""
    # 为了稳定，把缺失填 0，再转为字符串拼接
    arr = df[cols].fillna(0).values
    hashes = []
    for row in arr:
        s = ",".join(f"{x:.10g}" for x in row)  # 控制格式避免浮点噪声
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        hashes.append(h)
    return pd.Series(hashes, index=df.index)


def pick_hard_samples(pool_df: pd.DataFrame, y_pool: np.ndarray,
                      logits_pool_avg: np.ndarray,
                      per_class: int, prefer_wrong=True):
    """
    根据 avg logits 选择困难样本：
    - 先取“被 avg 误判”的样本（如 prefer_wrong=True）
    - 若不够，再按 margin（p_true - max_other）从小到大补齐
    """
    probs = softmax(logits_pool_avg)
    y_pred = probs.argmax(axis=1)
    # margin
    p_true = probs[np.arange(len(probs)), y_pool]
    probs[np.arange(len(probs)), y_pool] = -1.0  # 屏蔽真类以取最大错类概率
    p_other = probs.max(axis=1)
    margin = p_true - p_other

    chosen_idx = []
    for c in range(NUM_CLASSES):
        idx_c = np.where(y_pool == c)[0]
        if len(idx_c) == 0:
            continue
        # 该类中误判的索引
        wrong_idx = idx_c[y_pred[idx_c] != c]
        # 该类中按 margin 从小到大排序
        order = idx_c[np.argsort(margin[idx_c])]
        select = []

        if prefer_wrong and len(wrong_idx) > 0:
            # 先拿误判（最难）
            take = min(per_class, len(wrong_idx))
            # 对误判再按照 margin 排序，优先拿更难的
            w_sorted = wrong_idx[np.argsort(margin[wrong_idx])]
            select.extend(w_sorted[:take])

        # 不足则从整体难样本里补
        if len(select) < per_class:
            need = per_class - len(select)
            # 将 order 中未被选过的按顺序补齐
            for i in order:
                if i not in select:
                    select.append(i)
                    if len(select) >= per_class:
                        break
        chosen_idx.extend(select)

    return np.array(chosen_idx, dtype=np.int64), margin


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", type=str, default=TRAIN_DIR_DEFAULT)
    ap.add_argument("--test-path", type=str, default=TEST_PATH_DEFAULT)
    ap.add_argument("--out", type=str, default=OUT_PATH_DEFAULT)
    ap.add_argument("--pool-per-class", type=int, default=2000, help="候选池每类上限")
    ap.add_argument("--target-per-class", type=int, default=400, help="最终每类样本数")
    ap.add_argument("--max-clients", type=int, default=0, help="用于评估的客户端数量，0=用全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefer-wrong", action="store_true", help="优先选误判样本")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # 1) 特征规范
    feat_list, schema = ensure_feature_spec(args.clients_dir, args.test_path)

    # 2) 构建候选池（从所有客户端均衡抽样）
    client_files = [f for f in os.listdir(args.clients_dir) if f.lower().endswith(".csv")]
    client_files.sort()

    pool_parts = []
    per_class_cap = args.pool_per_class
    # 类别分桶
    cls_buckets = {i: [] for i in range(NUM_CLASSES)}

    print("==> 汇总候选样本池（分层抽样）...")
    for f in tqdm(client_files, ncols=80, unit="文件"):
        df = pd.read_csv(os.path.join(args.clients_dir, f), low_memory=False)
        if "Label" not in df.columns: 
            continue
        y = parse_label_series_7(df["Label"])
        mask = (y >= 0)
        if mask.sum() == 0: 
            continue
        df = df.loc[mask].reset_index(drop=True)
        y = y[mask]
        # 仅保留项目统一的数值特征
        df_num = df[feat_list + ["Label"]].copy()

        # 放入类桶
        for cid in range(NUM_CLASSES):
            idx = np.where(y == cid)[0]
            if len(idx) == 0:
                continue
            # 每次从该文件中对该类最多取若干条，防止某个文件压倒性占比
            take = min(len(idx), per_class_cap // max(1, len(client_files)//2) + 1)
            picked = np.random.choice(idx, size=min(len(idx), take), replace=False)
            cls_buckets[cid].append(df_num.iloc[picked])

    # 合并各类候选
    pool = []
    for cid in range(NUM_CLASSES):
        if len(cls_buckets[cid]) == 0:
            continue
        cat = pd.concat(cls_buckets[cid], axis=0, ignore_index=True)
        # 每类截断到 pool-per-class
        if len(cat) > per_class_cap:
            cat = cat.sample(n=per_class_cap, random_state=args.seed).reset_index(drop=True)
        pool.append(cat)
    pool_df = pd.concat(pool, axis=0, ignore_index=True)
    y_pool = parse_label_series_7(pool_df["Label"])
    pool_df = pool_df.drop(columns=["Label"]).reset_index(drop=True)
    print(f"候选样本池规模：{len(pool_df)}（每类上限 {args.pool_per_class}）")

    # 3) 训练用于“难样本判别”的客户端模型（可全 600，也可限制）
    n_clients = len(client_files) if args.max_clients <= 0 else min(args.max_clients, len(client_files))
    print(f"==> 训练客户端模型用于难样本评估：{n_clients} 台")
    workers = [
        Worker(user_idx=u, feature_list=feat_list, schema=schema, class_weight=None, target_class2id=CLASS2ID)
        for u in range(n_clients)
    ]
    for u in tqdm(range(n_clients), ncols=80, unit="终端"):
        workers[u].user_round_train_eval()

    # 4) 用 avg 在候选池上推理，得到 logits 的平均（作为难度度量依据）
    print("==> 候选池上推理（平均融合）...")
    sum_logits = np.zeros((len(pool_df), NUM_CLASSES), dtype=np.float64)
    for u in tqdm(range(n_clients), ncols=80, unit="模型"):
        raw = workers[u].predict_raw_df(pool_df)
        sum_logits += raw
    logits_pool_avg = sum_logits / max(1, n_clients)

    # 5) 选择困难样本：优先误判，其次 margin 最小
    print("==> 选择困难样本（每类各 {}）...".format(args.target_per_class))
    chosen_idx, margin = pick_hard_samples(pool_df, y_pool, logits_pool_avg,
                                           per_class=args.target_per_class,
                                           prefer_wrong=args.prefer_wrong)
    hard_df = pool_df.iloc[chosen_idx].copy()
    hard_df["Label"] = [ID2CLASS[i] for i in y_pool[chosen_idx]]

    # 6) 与测试集去重（严格相等的数值特征行）
    print("==> 去重检查（与 test.csv）...")
    test_df = pd.read_csv(args.test_path, low_memory=False)
    y_test = parse_label_series_7(test_df["Label"])
    test_df = test_df.loc[y_test >= 0].reset_index(drop=True)
    test_df_num = test_df[feat_list].copy()

    h_hard = row_hash(hard_df, feat_list)
    h_test = row_hash(test_df_num, feat_list)
    set_test = set(h_test.values.tolist())
    keep_mask = ~h_hard.isin(set_test)
    removed = int((~keep_mask).sum())
    if removed > 0:
        print(f"   * 去除与 test 重复样本：{removed} 条")

    hard_df = hard_df.loc[keep_mask].reset_index(drop=True)

    # 7) 输出与报告
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hard_df.to_csv(args.out, index=False)

    print("\n=== 构建完成（困难验证集） ===")
    print(f"输出文件: {args.out}")
    print(f"总样本数: {len(hard_df)}")
    print("类别分布：")
    counts = hard_df["Label"].value_counts()
    for c in TARGET_CLASSES:
        v = int(counts.get(c, 0))
        pct = v / max(1, len(hard_df)) * 100
        print(f"{c:<9}: {v:6d} ({pct:5.2f}%)")

    # 简单难度统计
    chosen_margin = margin[chosen_idx]
    print("\n难度统计（margin 越小越难）：")
    print(f"mean={chosen_margin.mean():.4f}  std={chosen_margin.std():.4f}  min={chosen_margin.min():.4f}  max={chosen_margin.max():.4f}")


if __name__ == "__main__":
    main()

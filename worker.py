# worker.py
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_DIR = "./data/clients_weighted"

TARGET_CLASSES = ["portmap", "syn", "mssql", "ldap", "netbios", "udp", "benign"]
TARGET_CLASSES = [c.lower() for c in TARGET_CLASSES]
CLS2ID = {c: i for i, c in enumerate(TARGET_CLASSES)}

def parse_label_7(series: pd.Series) -> np.ndarray:
    out = []
    for t in series:
        key = str(t).split("_")[-1].replace("-", "").lower()
        out.append(CLS2ID.get(key, -1))
    return np.array(out, dtype=np.int32)

# 与主程序一致的列名规范化
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

def read_csv_sanitized(path, **kwargs):
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return sanitize_and_drop_columns(df)

class Worker:
    def __init__(self, user_idx: int, feature_list, schema,
                 class_weight=None, target_class2id=None,
                 csv_reader=None, train_dir=None):
        self.user_idx = user_idx
        self.feature_list = feature_list
        self.schema = schema
        self.class_weight = class_weight
        self.target_class2id = target_class2id or CLS2ID
        self.csv_reader = csv_reader or read_csv_sanitized
        self.train_dir = train_dir or DEFAULT_TRAIN_DIR

        files = [f for f in os.listdir(self.train_dir) if f.lower().endswith(".csv")]
        files.sort()
        self.user_file = os.path.join(self.train_dir, files[self.user_idx])

        self.model = None
        self.params = {
            "objective": "multiclass",
            "num_class": len(self.target_class2id),
            "metric": "multi_logloss",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
        }

    def _load_user_data(self):
        df = self.csv_reader(self.user_file)
        if "Label" not in df.columns:
            raise RuntimeError(f"{self.user_file} 缺少 Label 列")

        keep = [c for c in self.feature_list if c in df.columns]
        X = df[keep].copy()
        y = parse_label_7(df["Label"])
        mask = (y >= 0)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask]

        for c in keep:
            X[c] = X[c].astype(self.schema.get(c, {}).get("dtype", "float64"))
            X[c] = X[c].fillna(self.schema.get(c, {}).get("fill", 0.0))

        X_tr, X_va, y_tr, y_va = train_test_split(
            X.values, y, test_size=0.2, random_state=2024, stratify=y
        )
        return X_tr, X_va, y_tr, y_va, keep

    def user_round_train_eval(self):
        X_tr, X_va, y_tr, y_va, cols = self._load_user_data()
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=cols, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, feature_name=cols, free_raw_data=False)
        self.model = lgb.train(
            params=self.params,
            train_set=dtr,
            valid_sets=[dva],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)],
        )
        yhat = np.argmax(self.model.predict(X_va, num_iteration=self.model.best_iteration), axis=1)
        return float((yhat == y_va).mean())

    def predict_raw_df(self, df_in: pd.DataFrame):
        df = df_in.copy()
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])
        df = sanitize_and_drop_columns(df)

        X = pd.DataFrame()
        for c in self.feature_list:
            if c in df.columns:
                X[c] = df[c].astype(self.schema.get(c, {}).get("dtype", "float64")).fillna(
                    self.schema.get(c, {}).get("fill", 0.0)
                )
            else:
                X[c] = self.schema.get(c, {}).get("fill", 0.0)
        # 返回 raw logits（未 softmax）
        return self.model.predict(X.values, num_iteration=self.model.best_iteration)

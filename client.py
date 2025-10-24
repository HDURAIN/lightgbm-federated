# -*- coding: utf-8 -*-
# client.py —— 客户端：本地训练与预测（LightGBM）

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional, List
from contextlib import contextmanager
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 客户端数据目录（与服务器端保持一致）
TRAIN_DATASETS_DIR = "./data/clients_weighted"

def _sorted_csv_list(data_dir: str):
    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(".csv") and "ds_store" not in f.lower()]
    files.sort()
    return files

def get_user_data(user_idx: int):
    fnames = _sorted_csv_list(TRAIN_DATASETS_DIR)
    if not (0 <= user_idx < len(fnames)):
        raise IndexError(f"user_idx {user_idx} out of range (0..{len(fnames)-1})")
    fname = fnames[user_idx]
    fpath = os.path.join(TRAIN_DATASETS_DIR, fname)
    data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
    # 统一清理列名两侧空格
    data.columns = [c.strip() for c in data.columns]
    return data, fname

@contextmanager
def _suppress_stdout():
    old_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout

def coerce_numeric_with_schema(df: pd.DataFrame, feature_list: List[str], schema: Dict[str, Dict]) -> pd.DataFrame:
    X = df.reindex(columns=feature_list, fill_value=0).copy()
    for c in feature_list:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        fillv = schema.get(c, {}).get("fill", 0.0)
        dtype = schema.get(c, {}).get("dtype", "float64")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan).fillna(fillv).astype(dtype)
    return X

class ClientWorker(object):
    """
    单客户端训练器：
    - 从本端CSV读取数据
    - 预处理（与服务器共享的FEATURE_LIST/SCHEMA对齐）
    - 本地训练 LightGBM
    - 暴露：验证准确率、对测试集的 raw_score/logits 预测
    """
    def __init__(
        self,
        user_idx: int,
        feature_list: List[str],
        schema: Dict[str, Dict],
        class_weight: Optional[Dict[int, float]],
        target_class2id: Dict[str, int],
    ):
        self.user_idx = user_idx
        self.data, self.fname = get_user_data(self.user_idx)
        self.feature_names = list(feature_list)
        self.schema = schema
        self.class_weight = class_weight
        assert target_class2id and len(target_class2id) >= 2, "target_class2id 不能为空"
        self.target_class2id = target_class2id
        self.num_classes = len(self.target_class2id)

        self.params = {
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": self.num_classes,
            "metric": "multi_logloss",
            "num_leaves": 40,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 40,
            "lambda_l2": 0.1,
            "verbosity": -1,
        }

        self.lgb_train, self.lgb_eval = self._preprocess_data()

    def _parse_labels_7(self, series: pd.Series) -> np.ndarray:
        out = []
        for t in series:
            # 更稳健：去空格、保留下划线后缀、去掉连接符与内部空格
            key = str(t).strip().split("_")[-1].replace("-", "").replace(" ", "").lower()
            out.append(self.target_class2id.get(key, -1))
        return np.array(out, dtype=np.int32).ravel()

    def _preprocess_data(self):
        label_series = (self.data["Label"] if "Label" in self.data.columns else self.data.iloc[:, -1])
        y_all = self._parse_labels_7(label_series)
        mask = (y_all >= 0)
        if not np.any(mask):
            raise ValueError(f"[Client {self.user_idx}] 本客户端在 7 类下没有样本，请检查数据划分/标签。")
        y = y_all[mask]
        X_all = coerce_numeric_with_schema(self.data, self.feature_names, self.schema)
        X = X_all.loc[mask].reset_index(drop=True)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )
        if self.class_weight is None:
            w_train = None
            w_val = None
        else:
            w_train = np.array([self.class_weight[int(c)] for c in self.train_y], dtype=np.float64)
            w_val   = np.array([self.class_weight[int(c)] for c in self.val_y],   dtype=np.float64)
        lgb_train = lgb.Dataset(self.train_x, self.train_y, weight=w_train)
        lgb_eval  = lgb.Dataset(self.val_x,   self.val_y,   reference=lgb_train, weight=w_val)
        return lgb_train, lgb_eval

    def train_and_eval(self) -> float:
        with _suppress_stdout():
            self.model = lgb.train(
                params=self.params,
                train_set=self.lgb_train,
                num_boost_round=300,
                valid_sets=[self.lgb_eval],
                valid_names=["valid_0"],
                callbacks=[
                    early_stopping(stopping_rounds=30, verbose=False),
                    log_evaluation(period=0),
                ],
            )
        preds = self.model.predict(self.val_x, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(self.val_y, y_pred)
        return float(acc)

    # 服务器在聚合阶段调用：
    def predict_raw_df(self, test_df: pd.DataFrame) -> np.ndarray:
        X = coerce_numeric_with_schema(test_df, self.feature_names, self.schema)
        return self.model.predict(X.values, num_iteration=self.model.best_iteration, raw_score=True)

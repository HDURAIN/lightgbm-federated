# -*- coding: utf-8 -*-
import os
import json
import argparse
import pandas as pd
import numpy as np

def numeric_cols(df: pd.DataFrame, drop_cols):
    cols = [c for c in df.columns if c not in drop_cols]
    # 仅用 dtype 排除 object；避免把 Label 当特征
    non_obj = df[cols].select_dtypes(exclude=["object"]).columns.tolist()
    # 防御：某些“看似数值”的列被读成 object，再试一次温和转数值探测
    fallback = []
    for c in cols:
        if c in non_obj:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # 若有效数值占比 >= 0.98，认为是数值列
        valid_ratio = np.isfinite(s).mean()
        if valid_ratio >= 0.98:
            fallback.append(c)
    return sorted(list(set(non_obj) | set(fallback)))

def main():
    ap = argparse.ArgumentParser(description="重建 feature_list.txt 与 schema.json（数值列交集）")
    ap.add_argument("--clients_dir", default="./data/clients_weighted", help="客户端CSV目录")
    ap.add_argument("--test_csv", default="./data/data_test/test.csv", help="测试集CSV路径")
    ap.add_argument("--feature_list", default="feature_list.txt", help="输出特征白名单路径")
    ap.add_argument("--schema", default="schema.json", help="输出schema路径")
    ap.add_argument("--drop_cols", default="Label,Unnamed: 0", help="在候选中特意排除的列（逗号分隔）")
    args = ap.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    # —— 收集客户端文件 —— 
    client_files = [f for f in os.listdir(args.clients_dir)
                    if f.lower().endswith(".csv") and "ds_store" not in f.lower()]
    client_files.sort()
    if not client_files:
        raise RuntimeError(f"在 {args.clients_dir} 未发现客户端CSV。")

    # —— 客户端特征并集（数值列）——
    client_union = set()
    for i, f in enumerate(client_files, 1):
        path = os.path.join(args.clients_dir, f)
        df = pd.read_csv(path, low_memory=False)
        # 容忍表头里偶发的空格
        df.columns = [c.strip() for c in df.columns]
        nc = set(numeric_cols(df, drop_cols))
        client_union = nc if i == 1 else (client_union | nc)

    # —— 测试集数值列 —— 
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    test_df.columns = [c.strip() for c in test_df.columns]
    test_numeric = set(numeric_cols(test_df, drop_cols))

    # —— 交集：既在测试集里也是数值、又在客户端并集中 —— 
    feature_set = sorted(list(client_union & test_numeric))
    if not feature_set:
        raise RuntimeError("客户端并集与测试集的数值特征交集为空，请检查列名/空格/类型。")

    # —— 写出 feature_list 与 schema —— 
    with open(args.feature_list, "w", encoding="utf-8") as f:
        for c in feature_set:
            f.write(c + "\n")

    schema = {c: {"dtype": "float64", "fill": 0.0} for c in feature_set}
    with open(args.schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # —— 汇报信息 —— 
    print("✅ 重建完成")
    print(f"客户端文件数：{len(client_files)}")
    print(f"客户端数值列并集：{len(client_union)}")
    print(f"测试集数值列数：{len(test_numeric)}")
    print(f"最终可用特征交集：{len(feature_set)} 列")
    print("前 10 列示例：", feature_set[:10])

if __name__ == "__main__":
    main()

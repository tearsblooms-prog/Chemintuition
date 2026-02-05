import pandas as pd
import json
import numpy as np


def process_predictions(file_path):
    try:
        # 1. 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"原始数据条数: {len(df)}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 2. 将 'predicted_fingerprint_json' 列从字符串解析为列表
    # 虽然这里可能不需要再做判断，但为了后续如果有其他分析，解析出来是好的实践
    if 'predicted_fingerprint_json' in df.columns:
        df['fingerprint_list'] = df['predicted_fingerprint_json'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    # 3. 过滤逻辑 (已移除)
    # 此步骤已移至 predict.py 的生成阶段。
    # 此时读取的 CSV 应该已经是过滤后的有效数据。

    # 直接使用 df 作为 df_filtered
    df_filtered = df

    # 检查是否有 predicted_yield 列
    if 'predicted_yield' not in df_filtered.columns:
        print("错误: CSV 文件中未包含 'predicted_yield' 列，无法进行统计。")
        return

    # 4. 统计 predicted_yield 的区间分布
    # 区间: <30, 30-50, 50-70, 70-80, 80-90, 90-100, >100
    # 定义分箱边界 (right=False: 左闭右开)
    bins = [-np.inf, 30, 50, 70, 80, 90, 100.000001, np.inf]

    # 定义标签
    labels = ['<30', '30-50', '50-70', '70-80', '80-90', '90-100', '>100']

    # 进行分箱
    df_filtered['yield_group'] = pd.cut(
        df_filtered['predicted_yield'],
        bins=bins,
        labels=labels,
        right=False
    )

    # 5. 统计结果
    distribution = df_filtered['yield_group'].value_counts().sort_index()

    print("\npredicted_yield 区间分布统计:")
    print(distribution)

    # 如果需要，可以将结果保存
    distribution.to_csv('yield_distribution.csv')
    print("分布统计已保存至 yield_distribution.csv")


# 执行函数
if __name__ == "__main__":
    # 请根据实际运行位置调整路径
    process_predictions('predictions_output.csv')
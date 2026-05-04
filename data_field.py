"""
data_field.py - 为CSV文件添加均线偏离度特征

计算并添加三个新列:
- m5:  (close - MA5)  / MA5   — 价格偏离5日均线
- m10: (close - MA10) / MA10  — 价格偏离10日均线
- m20: (close - MA20) / MA20  — 价格偏离20日均线

边缘处理：借鉴 data.py 中 amount/exchange 的方法，
当左侧数据不足时从右侧借用，保持完整的滑动窗口大小。

用法:
  python data_field.py                 # 处理 data/ 目录下所有CSV
  python data_field.py --dir data_all  # 处理指定目录
"""

import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

MA_WINDOWS = [5, 10, 20]
MA_COL_NAMES = ['m5', 'm10', 'm20']


def compute_ma_features(closes, window):
    """
    计算单只股票的均线偏离度

    Args:
        closes: 收盘价数组（时间正序）
        window: 均线窗口大小

    Returns:
        result: (close - MA) / MA 数组
    """
    n = len(closes)
    ma = np.empty(n, dtype=np.float64)

    # 向量化：累积和，用于快速计算滑动窗口均值
    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0
    np.cumsum(closes, out=cumsum[1:])

    # 正常情况: MA[i] = mean(closes[i-window:i])
    # 边缘情况: 左侧不足时从右侧借用（不含当天本身）
    for i in range(n):
        if i >= window:
            ma[i] = (cumsum[i] - cumsum[i - window]) / window
        else:
            left_part = closes[0:i]
            deficit = window - i
            right_end = min(i + 1 + deficit, n)
            right_part = closes[i + 1:right_end]
            combined = np.concatenate([left_part, right_part])
            if len(combined) > 0:
                ma[i] = np.mean(combined)
            else:
                ma[i] = closes[i]

    return np.where(ma > 0, (closes - ma) / ma, 0.0)


def process_single_file(args):
    """处理单个CSV文件，添加均线偏离度列"""
    file_path, = args
    try:
        df = pd.read_csv(file_path)

        # 已有全部列则跳过
        if all(col in df.columns for col in MA_COL_NAMES):
            return (file_path, 'skip')

        # 反转为时间正序（CSV默认倒序，最新在前）
        df = df.iloc[::-1].reset_index(drop=True)

        closes = df['close'].values.astype(np.float64)

        for window, col_name in zip(MA_WINDOWS, MA_COL_NAMES):
            df[col_name] = compute_ma_features(closes, window).astype(np.float32)

        # 反转回原始顺序并保存
        df = df.iloc[::-1].reset_index(drop=True)
        df.to_csv(file_path, index=False)
        return (file_path, 'ok')
    except Exception as e:
        return (file_path, f'error: {e}')


def main():
    parser = argparse.ArgumentParser(description='为CSV文件添加均线偏离度特征')
    parser.add_argument('--dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                        help='CSV文件目录（默认: data/）')
    args = parser.parse_args()

    data_dir = args.dir
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    if not csv_files:
        print(f"目录 {data_dir} 中没有CSV文件")
        return

    print(f"共 {len(csv_files)} 个CSV文件待处理")

    file_args = [(os.path.join(data_dir, f),) for f in csv_files]
    num_workers = min(cpu_count(), 8)

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, file_args)

    ok_count = sum(1 for _, status in results if status == 'ok')
    skip_count = sum(1 for _, status in results if status == 'skip')
    error_count = sum(1 for _, status in results if status.startswith('error'))

    print(f"完成: {ok_count} 个已处理, {skip_count} 个已跳过, {error_count} 个出错")

    if error_count > 0:
        print("\n错误详情:")
        for path, status in results:
            if status.startswith('error'):
                print(f"  {os.path.basename(path)}: {status}")


if __name__ == '__main__':
    main()

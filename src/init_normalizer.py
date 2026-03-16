"""
特征归一化器设置脚本

功能：
1. 在训练集上拟合特征归一化器
2. 保存归一化器到文件
3. 验证归一化效果

使用方法：
    python setup_feature_normalizer.py
"""

import os
import sys
import argparse

from data import load_and_preprocess_data, set_feature_normalizer, disable_feature_normalizer
from feature_normalizer import FeatureNormalizer


def main():
    # 设置工作目录为脚本所在目录（src/）
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='特征归一化器设置工具')
    parser.add_argument('--output-distribution', type=str, default='normal',
                        choices=['normal', 'uniform'],
                        help='输出分布类型: normal (标准正态) 或 uniform (均匀分布)')
    parser.add_argument('--n-quantiles', type=int, default=1000,
                        help='分位数数量（默认1000，越大越精确但越慢）')
    parser.add_argument('--output', type=str, default='./feature_normalizer.pkl',
                        help='输出文件路径')
    parser.add_argument('--force', action='store_true',
                        help='强制重新拟合，即使文件已存在')

    args = parser.parse_args()

    print("="*70)
    print("特征归一化器设置工具")
    print("="*70)

    # 检查文件是否已存在
    if os.path.exists(args.output) and not args.force:
        print(f"\n归一化器文件已存在: {args.output}")
        print("如需重新拟合，请使用 --force 参数")
        response = input("是否加载现有归一化器？(y/n): ")
        if response.lower() == 'y':
            normalizer = FeatureNormalizer.load(args.output)
            print("\n✓ 已加载现有归一化器")
            return

    print("\n[步骤1] 加载训练数据...")
    print("注意：归一化器只在训练集上拟合，避免数据泄漏")

    # 禁用归一化器以加载原始数据
    disable_feature_normalizer()

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print(f"\n训练集股票数: {len(train_stock_info)}")
    print(f"测试集股票数: {len(test_stock_info)}")

    print("\n[步骤2] 创建特征归一化器...")
    print(f"  输出分布: {args.output_distribution}")
    print(f"  分位数数量: {args.n_quantiles}")

    normalizer = FeatureNormalizer(
        output_distribution=args.output_distribution,
        n_quantiles=args.n_quantiles
    )

    print("\n[步骤3] 在训练集上拟合归一化器...")
    print("⏳ 这可能需要几分钟...")

    normalizer.fit(train_stock_info)

    print("\n[步骤4] 保存归一化器...")
    normalizer.save(args.output)

    print("\n[步骤5] 启用归一化器...")
    set_feature_normalizer(normalizer)

    print("\n" + "="*70)
    print("✓ 特征归一化器设置完成！")
    print("="*70)

    print("\n后续使用说明：")
    print(f"1. 归一化器已保存到: {args.output}")
    print("2. 在训练脚本中加载：")
    print("   from data import set_feature_normalizer")
    print("   from feature_normalizer import FeatureNormalizer")
    print("   normalizer = FeatureNormalizer.load('./feature_normalizer.pkl')")
    print("   set_feature_normalizer(normalizer)")
    print("\n3. 在模型中可以考虑移除 LayerNorm，因为预处理已做好")
    print("   如果保留 LayerNorm，会起到额外的稳定作用")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
EquiNet 数据维护工具 - 交互式菜单

统一入口，提供以下功能：
1. 更新数据（从外部数据源同步）
2. 筛选股票（全量池 → 训练池）
3. 检查数据质量（完整性验证与修复）
4. 计算特征（均线偏离度）
5. 数据库状态
6. 备份数据库
"""

import os
import sys

from data_maintenance.database import DatabaseManager


def _get_data_source():
    """从配置文件获取数据源类型"""
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from config import DataConfig
    return DataConfig.DATA_SOURCE


def _get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def print_header():
    print("\n" + "=" * 50)
    print("  EquiNet 数据维护工具")
    print("=" * 50)
    print(" 0. SQL 控制台    (手动执行 SQL)")
    print(" 1. 更新数据      (从外部数据源同步)")
    print(" 2. 筛选股票      (全量池 → 训练池)")
    print(" 3. 检查数据质量  (完整性验证与修复)")
    print(" 4. 计算特征      (均线偏离度)")
    print(" 5. 数据库状态")
    print(" 6. 备份数据库")
    print(" 7. 退出")
    print("=" * 50)


def handle_sql(db: DatabaseManager):
    """SQL 控制台"""
    print("\n--- SQL 控制台 ---")
    print("输入 SQL 语句执行，输入空行或 quit 返回主菜单")

    while True:
        try:
            sql = input("\nSQL> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not sql or sql.lower() in ('quit', 'exit', 'q'):
            break

        is_select = sql.upper().lstrip().startswith('SELECT')
        try:
            if is_select:
                cursor = db._conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                if columns:
                    print(' | '.join(columns))
                    print('-' * (len(' | '.join(columns))))
                    for row in rows[:50]:
                        print(' | '.join(str(v) for v in row))
                    if len(rows) > 50:
                        print(f"... 共 {len(rows)} 行 (仅显示前 50 行)")
                    elif rows:
                        print(f"({len(rows)} 行)")
                else:
                    print("（无结果）")
            else:
                cursor = db._conn.execute(sql)
                db._conn.commit()
                print(f"✓ 执行成功，影响 {cursor.rowcount} 行")
        except Exception as e:
            print(f"✗ 错误: {e}")


def handle_update(db: DatabaseManager):
    """更新数据"""
    print("\n--- 更新数据 ---")
    print("模式:")
    print("  1. 增量更新 (更新全量池已有股票)")
    print("  2. 全量更新 (拉取所有 A 股)")
    print("  3. 训练更新 (更新训练池股票，快速)")
    print("  0. 返回")

    choice = input("请选择模式 [1-3]: ").strip()
    mode_map = {'1': 'incremental', '2': 'full', '3': 'train'}
    mode = mode_map.get(choice)
    if mode is None:
        print("返回主菜单")
        return

    custom_stocks = input("指定股票代码 (留空=全部，空格分隔): ").strip()
    stock_codes = custom_stocks.split() if custom_stocks else None

    backup = input("是否备份？(Y/n): ").strip().lower()
    enable_backup = backup != 'n'

    data_source = _get_data_source()
    from data_maintenance.update import create_updater
    updater = create_updater(db, data_source, enable_backup)
    updater.update_all_stocks(mode, stock_codes)


def handle_select(db: DatabaseManager):
    """筛选股票"""
    print("\n--- 筛选股票 ---")

    dry_run = input("仅预览不修改？(y/N): ").strip().lower() == 'y'

    from data_maintenance.select import run_select
    selected = run_select(db, dry_run=dry_run)

    if dry_run:
        print(f"\n预览: 筛选出 {len(selected)} 只股票")
        if selected[:10]:
            print(f"  前10只: {', '.join(selected[:10])}")
            if len(selected) > 10:
                print(f"  ... 共 {len(selected)} 只")


def handle_check(db: DatabaseManager):
    """检查数据质量"""
    print("\n--- 检查数据质量 ---")

    pool_choice = input("检查池 (1=全量池all, 2=训练池selected) [1]: ").strip()
    pool_type = 'selected' if pool_choice == '2' else 'all'

    days_input = input(f"检查最近天数 [100]: ").strip()
    check_days = int(days_input) if days_input else 100

    verbose = input("详细输出？(y/N): ").strip().lower() == 'y'
    backup = input("检查前备份？(Y/n): ").strip().lower() != 'n'

    custom_stocks = input("指定股票 (留空=全部): ").strip()
    stock_codes = custom_stocks.split() if custom_stocks else None

    data_source = _get_data_source()
    from data_maintenance.check import create_checker
    checker = create_checker(db, data_source, check_days, backup)
    checker.run_full_check(stock_codes, pool_type, verbose)


def handle_features(db: DatabaseManager):
    """计算 MA 特征"""
    print("\n--- 计算特征 (均线偏离度) ---")

    pool_choice = input("目标池 (1=全量池, 2=训练池) [2]: ").strip()
    pool_type = 'all' if pool_choice == '1' else 'selected'

    force = input("强制重新计算？(y/N): ").strip().lower() == 'y'

    custom_stocks = input("指定股票 (留空=全部): ").strip()
    stock_codes = custom_stocks.split() if custom_stocks else None

    from data_maintenance.features import compute_features
    compute_features(db, pool_type, stock_codes, force)


def handle_status(db: DatabaseManager):
    """显示数据库状态"""
    print("\n--- 数据库状态 ---")

    stats = db.get_db_stats()
    print(f"数据库路径: {db.db_path}")
    print(f"数据库大小: {stats['db_size_mb']:.1f} MB")
    print(f"总行情数据: {stats['total_rows']:,} 条")
    print(f"总股票数:   {stats['total_stocks']}")
    print(f"全量池(all):      {stats['all_count']} 只")
    print(f"训练池(selected):  {stats['selected_count']} 只")
    print(f"缺特征股票:  {stats['stocks_missing_features']} 只")

    if stats['date_range'][0]:
        print(f"日期范围:   {stats['date_range'][0]} ~ {stats['date_range'][1]}")


def handle_backup(db: DatabaseManager):
    """备份数据库"""
    print("\n--- 备份数据库 ---")
    db.backup_database()


def main():
    """主交互循环"""
    db = DatabaseManager()

    try:
        while True:
            print_header()
            choice = input("请选择 [0-7]: ").strip()

            if choice == '0':
                handle_sql(db)
            elif choice == '1':
                handle_update(db)
            elif choice == '2':
                handle_select(db)
            elif choice == '3':
                handle_check(db)
            elif choice == '4':
                handle_features(db)
            elif choice == '5':
                handle_status(db)
            elif choice == '6':
                handle_backup(db)
            elif choice == '7':
                print("退出")
                break
            else:
                print("无效选择，请重试")

    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        db.close()


if __name__ == '__main__':
    main()

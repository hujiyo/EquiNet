"""
CSV → SQLite 迁移工具

将 data_all/ 和 data/ 目录下的 CSV 文件一次性导入到 SQLite 数据库。

迁移流程：
1. 验证数据库是否为空
2. 导入 data_all/*.csv → stock_daily（m5/m10/m20=NULL）+ stock_pool(all)
3. 导入 data/*.csv → 更新 stock_daily 的 m5/m10/m20 + stock_pool(selected)
4. 校验迁移结果
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from .database import DatabaseManager


class CSVMigrator:
    def __init__(self, db: DatabaseManager, project_root: str):
        self.db = db
        self.project_root = Path(project_root)
        self.data_all_dir = self.project_root / 'data_all'
        self.data_dir = self.project_root / 'data'

    def check_prerequisites(self) -> bool:
        """检查迁移前置条件"""
        if not self.data_all_dir.exists():
            print(f"✗ data_all/ 目录不存在: {self.data_all_dir}")
            return False

        stock_count = self.db.get_stock_count()
        if stock_count > 0:
            print(f"✗ 数据库中已有 {stock_count} 只股票的数据")
            print(f"  请清空数据库后重试，或使用 force=True 强制覆盖")
            return False

        csv_files = list(self.data_all_dir.glob("*.csv"))
        if not csv_files:
            print(f"✗ data_all/ 目录中没有 CSV 文件")
            return False

        print(f"✓ 迁移前检查通过:")
        print(f"  data_all/: {len(csv_files)} 个 CSV 文件")

        if self.data_dir.exists():
            data_csv = list(self.data_dir.glob("*.csv"))
            print(f"  data/: {len(data_csv)} 个 CSV 文件")

        return True

    def migrate_data_all(self) -> int:
        """迁移 data_all/ 目录的 CSV 文件到数据库（批量事务 + 向量化）"""
        csv_files = sorted(self.data_all_dir.glob("*.csv"))
        total = len(csv_files)
        print(f"\n[Step 1/3] 迁移 data_all/ ({total} 只股票)...")

        conn = self.db._conn

        # 迁移专用优化：关闭同步，大幅提升写入速度
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA cache_size=-512000")  # 512MB 缓存

        migrated = 0
        total_rows = 0
        pool_codes = []
        t0 = time.time()

        # 开启大事务
        conn.execute("BEGIN TRANSACTION")

        try:
            for i, csv_path in enumerate(csv_files, 1):
                stock_code = csv_path.stem
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        continue

                    df['date'] = df['date'].astype(int)

                    # 向量化构建 records，避免 iterrows
                    n = len(df)
                    codes = np.full(n, stock_code, dtype=object)
                    dates = df['date'].values.astype(np.int64).tolist()

                    # 各列转换为 float，NaN → None
                    def to_float_arr(series):
                        arr = series.values.astype(np.float64)
                        mask = np.isnan(arr)
                        arr_list = arr.tolist()
                        for j in range(n):
                            if mask[j]:
                                arr_list[j] = None
                        return arr_list

                    opens = to_float_arr(df['open'])
                    highs = to_float_arr(df['high'])
                    lows = to_float_arr(df['low'])
                    closes = to_float_arr(df['close'])
                    amounts = df['amount'].fillna(0.0).values.tolist()
                    volumes = df['volume'].fillna(0.0).values.tolist()
                    exchanges = df['exchange'].fillna(0.0).values.tolist()

                    # vwap 处理 NaN
                    vwap_arr = df['vwap'].values.astype(np.float64)
                    vwap_mask = np.isnan(vwap_arr)
                    vwaps = vwap_arr.tolist()
                    for j in range(n):
                        if vwap_mask[j]:
                            vwaps[j] = None

                    # m5/m10/m20 全部为 None
                    nones = [None] * n

                    records = list(zip(codes, dates, opens, highs, lows, closes,
                                       amounts, volumes, exchanges, vwaps,
                                       nones, nones, nones))

                    conn.executemany(
                        "INSERT OR REPLACE INTO stock_daily "
                        "(stock_code, date, open, high, low, close, amount, volume, exchange, vwap, m5, m10, m20) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        records
                    )

                    pool_codes.append(stock_code)
                    migrated += 1
                    total_rows += n

                    if i % 200 == 0 or i == total:
                        elapsed = time.time() - t0
                        speed = total_rows / elapsed if elapsed > 0 else 0
                        print(f"  [{i}/{total}] 已迁移 {migrated} 只 ({total_rows:,} 行, {speed:,.0f} 行/秒)")

                except Exception as e:
                    print(f"  ✗ {stock_code} 迁移失败: {e}")

            conn.commit()

            # 批量写入 pool
            if pool_codes:
                conn.executemany(
                    "INSERT OR IGNORE INTO stock_pool (stock_code, pool_type) VALUES (?, 'all')",
                    [(code,) for code in pool_codes]
                )
                conn.commit()

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"✗ 迁移失败，已回滚: {e}")
            return 0

        # 恢复正常 PRAGMA
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA journal_mode=WAL")

        elapsed = time.time() - t0
        print(f"✓ data_all/ 迁移完成: {migrated}/{total} 只 ({total_rows:,} 行), 耗时 {elapsed:.1f}s")
        return migrated

    def migrate_data(self) -> int:
        """迁移 data/ 目录的 CSV 文件，更新 m5/m10/m20 特征并标记为 selected 池"""
        if not self.data_dir.exists():
            print("⚠ data/ 目录不存在，跳过 Step 2")
            return 0

        csv_files = sorted(self.data_dir.glob("*.csv"))
        total = len(csv_files)
        if total == 0:
            print("⚠ data/ 目录中没有 CSV 文件，跳过 Step 2")
            return 0

        print(f"\n[Step 2/3] 迁移 data/ ({total} 只股票) - 更新特征列...")

        conn = self.db._conn
        conn.execute("PRAGMA synchronous=OFF")

        migrated = 0
        selected_codes = []
        t0 = time.time()

        conn.execute("BEGIN TRANSACTION")

        try:
            for i, csv_path in enumerate(csv_files, 1):
                stock_code = csv_path.stem
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        continue

                    df['date'] = df['date'].astype(int)

                    has_features = all(c in df.columns for c in ['m5', 'm10', 'm20'])
                    if has_features:
                        # 向量化构建 UPDATE 参数
                        dates = df['date'].values.astype(np.int64).tolist()
                        m5 = df['m5'].values.tolist()
                        m10 = df['m10'].values.tolist()
                        m20 = df['m20'].values.tolist()

                        # NaN → None
                        for arr in [m5, m10, m20]:
                            for j in range(len(arr)):
                                if arr[j] != arr[j]:  # NaN check
                                    arr[j] = None

                        update_records = list(zip(m5, m10, m20,
                                                  [stock_code] * len(dates), dates))

                        conn.executemany(
                            "UPDATE stock_daily SET m5=?, m10=?, m20=? "
                            "WHERE stock_code=? AND date=?",
                            update_records
                        )

                    selected_codes.append(stock_code)
                    migrated += 1

                    if i % 200 == 0 or i == total:
                        elapsed = time.time() - t0
                        print(f"  [{i}/{total}] 已处理 {migrated} 只 ({elapsed:.1f}s)")

                except Exception as e:
                    print(f"  ✗ {stock_code} 处理失败: {e}")

            conn.commit()

            if selected_codes:
                conn.executemany(
                    "INSERT OR IGNORE INTO stock_pool (stock_code, pool_type) VALUES (?, 'selected')",
                    [(code,) for code in selected_codes]
                )
                conn.commit()

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"✗ 迁移失败，已回滚: {e}")
            return 0

        conn.execute("PRAGMA synchronous=NORMAL")

        elapsed = time.time() - t0
        print(f"✓ data/ 迁移完成: {migrated}/{total} 只, 耗时 {elapsed:.1f}s")
        return migrated

    def verify_migration(self) -> bool:
        """校验迁移结果（快速抽样）"""
        print(f"\n[Step 3/3] 校验迁移结果...")

        stats = self.db.get_db_stats()
        print(f"  数据库总行数: {stats['total_rows']:,}")
        print(f"  股票数量: {stats['total_stocks']}")
        print(f"  全量池(all): {stats['all_count']} 只")
        print(f"  训练池(selected): {stats['selected_count']} 只")
        print(f"  日期范围: {stats['date_range'][0]} ~ {stats['date_range'][1]}")

        # 抽样校验：20 只股票比较行数
        sample_codes = self.db.get_pool_stocks('all')[:20]
        all_ok = True
        for code in sample_codes:
            db_rows = self.db.get_daily_row_count(code)
            csv_path = self.data_all_dir / f"{code}.csv"
            if csv_path.exists():
                csv_rows = len(pd.read_csv(csv_path))
                if db_rows != csv_rows:
                    print(f"  ✗ {code}: DB={db_rows}, CSV={csv_rows} (不一致)")
                    all_ok = False

        if all_ok:
            print(f"✓ 校验通过，所有抽样股票数据一致")
        else:
            print(f"⚠ 部分校验不一致，请检查上述输出")

        return all_ok

    def run(self, force=False):
        """执行完整迁移流程"""
        if force and self.db.get_stock_count() > 0:
            print("⚠ 强制模式：将覆盖已有数据")
            self.db._conn.execute("DELETE FROM stock_daily")
            self.db._conn.execute("DELETE FROM stock_pool")
            self.db._conn.commit()

        if not force and not self.check_prerequisites():
            return False

        t0 = time.time()

        self.migrate_data_all()
        self.migrate_data()
        self.verify_migration()

        # 重建索引统计
        self.db._conn.execute("ANALYZE")
        self.db._conn.commit()

        elapsed = time.time() - t0
        stats = self.db.get_db_stats()
        print(f"\n{'='*50}")
        print(f"迁移完成！耗时 {elapsed:.1f}s")
        print(f"数据库大小: {stats['db_size_mb']:.1f} MB")
        print(f"{'='*50}")
        return True

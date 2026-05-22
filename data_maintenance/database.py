"""
SQLite 数据库管理模块

提供 EquiNet 项目的所有数据库操作：
- Schema 初始化和版本管理
- 行情数据的 CRUD 操作
- 股票池管理（all / selected）
- 股票元数据管理
- 数据库备份（SQLite 内置 backup API）
"""

import os
import sqlite3
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatabaseManager:
    """EquiNet SQLite 数据库管理器"""

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS stock_daily (
        stock_code  TEXT    NOT NULL,
        date        INTEGER NOT NULL,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL    NOT NULL,
        amount      REAL    DEFAULT 0.0,
        volume      REAL    DEFAULT 0.0,
        exchange    REAL    DEFAULT 0.0,
        vwap        REAL,
        m5          REAL,
        m10         REAL,
        m20         REAL,
        dif         REAL,
        dea         REAL,
        macd_hist   REAL,
        bb_upper    REAL,
        bb_lower    REAL,
        updated_at  TEXT    DEFAULT (datetime('now')),
        PRIMARY KEY (stock_code, date)
    );

    CREATE INDEX IF NOT EXISTS idx_daily_date ON stock_daily (date);

    CREATE TABLE IF NOT EXISTS stock_pool (
        stock_code  TEXT    NOT NULL,
        pool_type   TEXT    NOT NULL,
        added_date  TEXT    DEFAULT (datetime('now')),
        is_active   INTEGER DEFAULT 1,
        PRIMARY KEY (stock_code, pool_type)
    );

    CREATE TABLE IF NOT EXISTS stock_metadata (
        stock_code      TEXT PRIMARY KEY,
        stock_name      TEXT,
        market          TEXT,
        is_st           INTEGER DEFAULT 0,
        market_cap      REAL,
        cap_updated_at  TEXT,
        updated_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS schema_info (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = self._default_db_path()
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    @staticmethod
    def _default_db_path():
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(pkg_dir, 'equinet.db')

    def init_schema(self):
        self._conn.executescript(self.SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_info (key, value) VALUES (?, ?)",
            ('version', '1')
        )

        # 增量迁移：添加 MACD 特征列
        existing_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(stock_daily)")}
        for col in ('dif', 'dea', 'macd_hist', 'bb_upper', 'bb_lower'):
            if col not in existing_cols:
                self._conn.execute(f"ALTER TABLE stock_daily ADD COLUMN {col} REAL")

        self._conn.commit()

    # ==================== 行情数据写入 ====================

    def upsert_daily_data(self, stock_code: str, df: pd.DataFrame):
        """将 DataFrame 格式的行情数据写入数据库（UPSERT），向量化构建"""
        cols = ['open', 'high', 'low', 'close', 'amount', 'volume',
                'exchange', 'vwap', 'm5', 'm10', 'm20', 'dif', 'dea', 'macd_hist',
                'bb_upper', 'bb_lower']

        n = len(df)
        codes = [stock_code] * n
        dates = df['date'].values.astype(np.int64).tolist()

        col_data = {}
        for c in cols:
            if c in df.columns:
                arr = df[c].values.astype(np.float64)
                obj_arr = arr.astype(object)
                obj_arr[np.isnan(arr)] = None
                col_data[c] = obj_arr.tolist()
            else:
                col_data[c] = [None] * n

        records = list(zip(codes, dates,
                           * [col_data[c] for c in cols]))

        self._conn.executemany(
            f"INSERT OR REPLACE INTO stock_daily "
            f"(stock_code, date, {', '.join(cols)}) "
            f"VALUES (?, ?, {', '.join(['?'] * len(cols))})",
            records
        )
        self._conn.commit()

    def bulk_upsert_daily(self, records: List[tuple]):
        """批量写入行情数据。records: [(stock_code, date, open, high, ...), ...]"""
        cols = ['stock_code', 'date', 'open', 'high', 'low', 'close',
                'amount', 'volume', 'exchange', 'vwap', 'm5', 'm10', 'm20',
                'dif', 'dea', 'macd_hist', 'bb_upper', 'bb_lower']
        placeholders = ', '.join(['?'] * len(cols))
        col_str = ', '.join(cols)
        self._conn.executemany(
            f"INSERT OR REPLACE INTO stock_daily ({col_str}) VALUES ({placeholders})",
            records
        )
        self._conn.commit()

    def update_features(self, stock_code: str, feature_records: List[tuple]):
        """更新指定股票的衍生特征。
        feature_records: [(date, m5, m10, m20, dif, dea, macd_hist, bb_upper, bb_lower), ...]"""
        self._conn.executemany(
            "UPDATE stock_daily SET m5=?, m10=?, m20=?, dif=?, dea=?, macd_hist=?, bb_upper=?, bb_lower=? "
            "WHERE stock_code=? AND date=?",
            [(m5, m10, m20, dif, dea, macd_hist, bb_upper, bb_lower, stock_code, date)
             for date, m5, m10, m20, dif, dea, macd_hist, bb_upper, bb_lower in feature_records]
        )
        self._conn.commit()

    def delete_daily_data(self, stock_code: str, start_date=None, end_date=None):
        """删除指定股票的行情数据"""
        if start_date and end_date:
            self._conn.execute(
                "DELETE FROM stock_daily WHERE stock_code=? AND date BETWEEN ? AND ?",
                (stock_code, start_date, end_date)
            )
        elif start_date:
            self._conn.execute(
                "DELETE FROM stock_daily WHERE stock_code=? AND date >= ?",
                (stock_code, start_date)
            )
        elif end_date:
            self._conn.execute(
                "DELETE FROM stock_daily WHERE stock_code=? AND date <= ?",
                (stock_code, end_date)
            )
        else:
            self._conn.execute("DELETE FROM stock_daily WHERE stock_code=?", (stock_code,))
        self._conn.commit()

    # ==================== 行情数据查询 ====================

    def get_stock_codes(self, pool_type=None) -> List[str]:
        """获取股票代码列表。pool_type='all'/'selected'/None(全部)"""
        if pool_type:
            cursor = self._conn.execute(
                "SELECT stock_code FROM stock_pool WHERE pool_type=? AND is_active=1 "
                "ORDER BY stock_code",
                (pool_type,)
            )
        else:
            cursor = self._conn.execute(
                "SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code"
            )
        return [row[0] for row in cursor.fetchall()]

    def get_stock_data(self, stock_code: str, start_date=None, end_date=None,
                       columns=None, chronological=True) -> pd.DataFrame:
        """
        查询单只股票的行情数据

        Args:
            stock_code: 股票代码
            start_date: 起始日期(YYYYMMDD)，None 表示不限制
            end_date: 结束日期(YYYYMMDD)，None 表示不限制
            columns: 指定列名列表，None 表示全部列
            chronological: True=正序(旧→新)，False=倒序(新→旧)

        Returns:
            DataFrame，包含 date 列和请求的数据列
        """
        col_str = ', '.join(columns) if columns else 'date, open, high, low, close, amount, volume, exchange, vwap, m5, m10, m20, dif, dea, macd_hist, bb_upper, bb_lower'

        conditions = ["stock_code = ?"]
        params = [stock_code]

        if start_date is not None:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("date <= ?")
            params.append(end_date)

        order = "ASC" if chronological else "DESC"
        query = f"SELECT {col_str} FROM stock_daily WHERE {' AND '.join(conditions)} ORDER BY date {order}"

        return pd.read_sql_query(query, self._conn, params=params)

    def get_latest_date(self, stock_code: str) -> Optional[int]:
        """获取指定股票的最新交易日期"""
        cursor = self._conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE stock_code=?", (stock_code,)
        )
        result = cursor.fetchone()[0]
        return int(result) if result is not None else None

    def get_date_range(self, stock_code: str) -> Tuple[Optional[int], Optional[int]]:
        """获取指定股票的日期范围 (最早, 最新)"""
        cursor = self._conn.execute(
            "SELECT MIN(date), MAX(date) FROM stock_daily WHERE stock_code=?",
            (stock_code,)
        )
        row = cursor.fetchone()
        return (int(row[0]) if row[0] else None, int(row[1]) if row[1] else None)

    def get_stock_count(self, pool_type=None) -> int:
        """获取股票数量"""
        if pool_type:
            cursor = self._conn.execute(
                "SELECT COUNT(DISTINCT stock_code) FROM stock_pool WHERE pool_type=? AND is_active=1",
                (pool_type,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_daily")
        return cursor.fetchone()[0]

    def get_daily_row_count(self, stock_code: str) -> int:
        """获取指定股票的行情数据行数"""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM stock_daily WHERE stock_code=?", (stock_code,)
        )
        return cursor.fetchone()[0]

    def get_stocks_missing_features(self, pool_type='selected') -> List[str]:
        """获取指定池中缺少特征（m5/m10/m20 或 MACD）的股票"""
        cursor = self._conn.execute(
            "SELECT sp.stock_code FROM stock_pool sp "
            "JOIN stock_daily sd ON sp.stock_code = sd.stock_code "
            "WHERE sp.pool_type=? AND sp.is_active=1 "
            "AND (sd.m5 IS NULL OR sd.dif IS NULL OR sd.bb_upper IS NULL) "
            "GROUP BY sp.stock_code",
            (pool_type,)
        )
        return [row[0] for row in cursor.fetchall()]

    # ==================== 股票池管理 ====================

    def add_to_pool(self, stock_codes: List[str], pool_type: str):
        """将股票添加到指定池"""
        records = [(code, pool_type) for code in stock_codes]
        self._conn.executemany(
            "INSERT OR IGNORE INTO stock_pool (stock_code, pool_type) VALUES (?, ?)",
            records
        )
        self._conn.commit()

    def remove_from_pool(self, stock_codes: List[str], pool_type: str):
        """从指定池中移除股票"""
        for code in stock_codes:
            self._conn.execute(
                "UPDATE stock_pool SET is_active=0 WHERE stock_code=? AND pool_type=?",
                (code, pool_type)
            )
        self._conn.commit()

    def get_pool_stocks(self, pool_type: str) -> List[str]:
        """获取指定池中的活跃股票列表"""
        cursor = self._conn.execute(
            "SELECT stock_code FROM stock_pool WHERE pool_type=? AND is_active=1 ORDER BY stock_code",
            (pool_type,)
        )
        return [row[0] for row in cursor.fetchall()]

    def sync_pool(self, stock_codes: List[str], pool_type: str):
        """增量同步：将指定列表设为池的活跃成员，不在列表中的设为非活跃"""
        current = set(self.get_pool_stocks(pool_type))
        target = set(stock_codes)

        to_add = target - current
        to_remove = current - target

        if to_add:
            self.add_to_pool(list(to_add), pool_type)
        if to_remove:
            self.remove_from_pool(list(to_remove), pool_type)

        return len(to_add), len(to_remove)

    def clear_pool(self, pool_type: str):
        """清空指定池"""
        self._conn.execute("DELETE FROM stock_pool WHERE pool_type=?", (pool_type,))
        self._conn.commit()

    # ==================== 元数据管理 ====================

    def upsert_metadata(self, stock_code: str, **kwargs):
        """写入或更新单只股票的元数据"""
        if not kwargs:
            return
        cols = list(kwargs.keys())
        vals = [kwargs[c] for c in cols]
        col_str = ', '.join(cols)
        placeholder_str = ', '.join(['?'] * len(cols))
        update_str = ', '.join(f"{c}=excluded.{c}" for c in cols)

        self._conn.execute(
            f"INSERT INTO stock_metadata (stock_code, {col_str}) VALUES (?, {placeholder_str}) "
            f"ON CONFLICT(stock_code) DO UPDATE SET {update_str}",
            [stock_code] + vals
        )
        self._conn.commit()

    def batch_upsert_metadata(self, records: List[Dict]):
        """批量写入元数据"""
        if not records:
            return
        cols = [k for k in records[0].keys() if k != 'stock_code']
        for rec in records:
            vals = [rec.get(c) for c in cols]
            col_str = ', '.join(cols)
            placeholder_str = ', '.join(['?'] * len(cols))
            update_str = ', '.join(f"{c}=excluded.{c}" for c in cols)
            self._conn.execute(
                f"INSERT INTO stock_metadata (stock_code, {col_str}) VALUES (?, {placeholder_str}) "
                f"ON CONFLICT(stock_code) DO UPDATE SET {update_str}",
                [rec['stock_code']] + vals
            )
        self._conn.commit()

    def get_metadata(self, stock_code: str) -> Optional[Dict]:
        """获取单只股票的元数据"""
        cursor = self._conn.execute(
            "SELECT * FROM stock_metadata WHERE stock_code=?", (stock_code,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    # ==================== 备份 ====================

    def backup_database(self, backup_path=None) -> str:
        """使用 SQLite 内置 backup API 创建数据库备份"""
        if backup_path is None:
            db_dir = Path(self.db_path).parent / 'backup'
            db_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(db_dir / f"equinet_{timestamp}.db")

        target_conn = sqlite3.connect(backup_path)
        try:
            self._conn.backup(target_conn)
        finally:
            target_conn.close()

        size_mb = os.path.getsize(backup_path) / 1024 / 1024
        print(f"✓ 数据库已备份到: {backup_path} ({size_mb:.1f} MB)")
        return backup_path

    # ==================== 统计信息 ====================

    def get_db_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}

        cursor = self._conn.execute("SELECT COUNT(*) FROM stock_daily")
        stats['total_rows'] = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_daily")
        stats['total_stocks'] = cursor.fetchone()[0]

        for pool_type in ['all', 'selected']:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM stock_pool WHERE pool_type=? AND is_active=1",
                (pool_type,)
            )
            stats[f'{pool_type}_count'] = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT COUNT(DISTINCT sd.stock_code) FROM stock_daily sd "
            "JOIN stock_pool sp ON sd.stock_code = sp.stock_code "
            "WHERE sp.pool_type='selected' AND sp.is_active=1 AND (sd.m5 IS NULL OR sd.dif IS NULL OR sd.bb_upper IS NULL)"
        )
        stats['stocks_missing_features'] = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT MIN(date), MAX(date) FROM stock_daily")
        row = cursor.fetchone()
        stats['date_range'] = (row[0], row[1])

        stats['db_size_mb'] = os.path.getsize(self.db_path) / 1024 / 1024

        return stats

    # ==================== 生命周期 ====================

    def close(self):
        conn = getattr(self, '_conn', None)
        if conn:
            conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

"""
股票数据筛选模块

从全量池中筛选符合条件的股票到训练池。

筛选条件：
- 主板股票（沪市 600/601/603/605 + 深市 000/001/002/003）
- 排除 ST/*ST 股票
- 排除已退市或长期停牌（最新数据超过30天）
- 市值筛选：流通市值在 [10亿, 200亿] 范围
"""

import os
import sys
import time
import datetime
from typing import List, Optional, Dict
import pandas as pd

from .database import DatabaseManager


class DataSelector:
    """从全量股票池中筛选符合条件的股票"""

    def __init__(self, db: DatabaseManager,
                 market_cap_min: float = None,
                 market_cap_max: float = None,
                 valid_prefixes: list = None):
        self.db = db

        # 从 config 读取默认值
        if market_cap_min is None or market_cap_max is None or valid_prefixes is None:
            src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from config import DataConfig
            self.market_cap_min = market_cap_min if market_cap_min is not None else DataConfig.MARKET_CAP_MIN
            self.market_cap_max = market_cap_max if market_cap_max is not None else DataConfig.MARKET_CAP_MAX
            self.valid_prefixes = valid_prefixes or DataConfig.VALID_STOCK_PREFIXES
        else:
            self.market_cap_min = market_cap_min
            self.market_cap_max = market_cap_max
            self.valid_prefixes = valid_prefixes

    def _get_all_stock_codes(self) -> List[str]:
        return self.db.get_pool_stocks('all')

    def _filter_by_code(self, stock_codes: List[str]) -> List[str]:
        """按代码前缀过滤，只保留主板股票"""
        filtered = [code for code in stock_codes
                    if any(code.startswith(prefix) for prefix in self.valid_prefixes)]
        excluded = len(stock_codes) - len(filtered)
        print(f"  [前缀过滤] {len(stock_codes)} → {len(filtered)} (排除 {excluded} 只创业板/科创板/北交所)")
        return filtered

    def _filter_by_active(self, stock_codes: List[str], max_days_behind: int = 30) -> List[str]:
        """排除退市或长期停牌股票（批量查询）"""
        today = datetime.datetime.now()
        all_latest = self.db.get_all_latest_dates()
        active = []
        inactive_count = 0

        for code in stock_codes:
            latest_date = all_latest.get(code)
            if latest_date is None:
                inactive_count += 1
                continue
            try:
                date_obj = datetime.datetime.strptime(str(latest_date), "%Y%m%d")
                if (today - date_obj).days <= max_days_behind:
                    active.append(code)
                else:
                    inactive_count += 1
            except ValueError:
                inactive_count += 1

        print(f"  [活跃过滤] {len(stock_codes)} → {len(active)} (排除 {inactive_count} 只退市/长期停牌)")
        return active

    def _get_stock_names(self) -> Dict[str, str]:
        """从 AKShare 获取所有 A 股股票名称（用于判断 ST）"""
        print("  [AKShare] 正在获取A股股票名称...")
        start_time = time.time()

        try:
            import akshare as ak
        except ImportError:
            print("  ✗ 请先安装 akshare: pip install akshare")
            return {}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_spot()
                elapsed = time.time() - start_time
                print(f"  [AKShare] 获取成功，共 {len(df)} 只股票，耗时 {elapsed:.1f} 秒")

                code_to_name = {}
                for _, row in df.iterrows():
                    raw_code = str(row['代码'])
                    if raw_code.startswith('bj'):
                        continue
                    # AKShare 返回带交易所前缀（sh600053 / sz000001），取后 6 位纯数字
                    code = raw_code[-6:]
                    name = str(row['名称'])
                    code_to_name[code] = name

                return code_to_name
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  [AKShare] 获取失败 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  [AKShare] 获取失败: {e}")
                    return {}

    def _batch_market_caps(self, stock_codes: List[str]) -> Dict[str, float]:
        """批量计算流通市值：成交额 × 100 / 换手率（一次 SQL）"""
        latest_df = self.db.get_latest_records_batch(stock_codes)
        if latest_df.empty:
            return {}
        valid = latest_df['exchange'].notna() & (latest_df['exchange'] > 0) & latest_df['amount'].notna()
        valid_df = latest_df[valid].copy()
        valid_df['market_cap'] = valid_df['amount'] * 100 / valid_df['exchange']
        return dict(zip(valid_df['stock_code'], valid_df['market_cap']))

    def select(self) -> List[str]:
        """执行完整的筛选流程"""
        all_codes = self._get_all_stock_codes()
        if not all_codes:
            print("✗ 全量池中没有股票数据")
            return []

        min_yi = self.market_cap_min / 1e8
        max_yi = self.market_cap_max / 1e8

        print(f"\n{'=' * 50}")
        print(f"股票数据筛选")
        print(f"{'=' * 50}")
        print(f"全量池：{len(all_codes)} 只")
        print(f"市值范围：{min_yi:.0f}亿 ~ {max_yi:.0f}亿（流通市值）")
        print(f"{'=' * 50}\n")

        # Step 1: 代码前缀过滤
        codes = self._filter_by_code(all_codes)

        # Step 2: 活跃度过滤
        codes = self._filter_by_active(codes)

        # Step 3: 市值 + ST 过滤
        code_to_name = self._get_stock_names()
        market_caps = self._batch_market_caps(codes)
        filtered = []
        st_count = 0
        under_cap_count = 0
        over_cap_count = 0
        no_data_count = 0

        for code in codes:
            name = code_to_name.get(code, '')
            if 'ST' in name.upper():
                st_count += 1
                continue

            market_cap = market_caps.get(code)
            if market_cap is None:
                no_data_count += 1
                continue
            if market_cap < self.market_cap_min:
                under_cap_count += 1
                continue
            if market_cap > self.market_cap_max:
                over_cap_count += 1
                continue

            filtered.append(code)

        print(f"  [市值+ST过滤] {len(codes)} → {len(filtered)} "
              f"(排除 {st_count} 只ST, {under_cap_count} 只市值<{min_yi:.0f}亿, "
              f"{over_cap_count} 只市值>{max_yi:.0f}亿, {no_data_count} 只无数据)")

        print(f"\n{'=' * 50}")
        print(f"筛选完成：{len(all_codes)} → {len(filtered)} 只")
        print(f"{'=' * 50}")

        return filtered

    def apply_selection(self, selected_codes: List[str]) -> int:
        """将筛选结果应用到 selected 池"""
        added, removed = self.db.sync_pool(selected_codes, 'selected')
        print(f"\n✓ 训练池已更新: +{added} 只, -{removed} 只, 当前 {len(selected_codes)} 只")
        return len(selected_codes)


def run_select(db: DatabaseManager, dry_run: bool = False,
               market_cap_min: float = None, market_cap_max: float = None) -> List[str]:
    """执行股票筛选"""
    selector = DataSelector(db, market_cap_min, market_cap_max)
    selected = selector.select()

    if not dry_run:
        selector.apply_selection(selected)
    else:
        print(f"\n[Dry Run] 未修改数据库")

    return selected

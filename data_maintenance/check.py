"""
股票数据质量检查模块

股票数据质量检查模块
基于外部数据源验证本地数据的完整性和准确性。

检查项：
- 数据时效性（是否滞后）
- 缺失交易日
- OHLC 逻辑正确性
- 价格异常波动
- 自动修复检测到的问题
"""

import time
import sys
import datetime
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .database import DatabaseManager
from .features import compute_features_for_stock
from .utils import normalize_stock_df


class CheckStatus(Enum):
    PASS = "✓"
    FAIL = "✗ 失败"
    WARNING = "⚠ 警告"
    SKIP = "○ 跳过"


@dataclass
class CheckResult:
    stock_code: str
    status: CheckStatus
    message: str
    details: Optional[Dict] = None


class DataChecker:
    """基于 Baostock 的数据质量检查器"""

    def __init__(self, db: DatabaseManager, check_days: int = 100, backup: bool = True):
        self.db = db
        self.check_days = check_days
        self.enable_backup = backup
        self.login_success = False
        self.stats = {'total': 0, 'pass': 0, 'fail': 0, 'warning': 0, 'skip': 0}

    def login(self) -> bool:
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == '0':
                self.login_success = True
                self.bs = bs
                return True
            else:
                print(f"✗ Baostock 登录失败：{lg.error_msg}")
                return False
        except Exception as e:
            print(f"✗ Baostock 登录异常：{e}")
            return False

    def logout(self):
        if self.login_success:
            self.bs.logout()
            self.login_success = False

    def fetch_external_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从 Baostock 获取指定日期范围的数据"""
        try:
            code_with_prefix = self._format_stock_code(stock_code)
            if code_with_prefix is None:
                return None

            rs = self.bs.query_history_k_data_plus(
                code_with_prefix,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3"
            )

            if rs.error_code != '0':
                return None

            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)
            return normalize_stock_df(df, source='baostock')
        except Exception as e:
            print(f"获取 Baostock 数据失败：{e}")
            return None

    def check_data_integrity(self, stock_code: str, verbose: bool = False) -> CheckResult:
        """检查单只股票的数据完整性"""
        latest_date = self.db.get_latest_date(stock_code)
        if latest_date is None:
            return CheckResult(stock_code, CheckStatus.FAIL, "数据库中无数据")

        today = datetime.datetime.now()
        date_obj = datetime.datetime.strptime(str(latest_date), "%Y%m%d")
        days_diff = (today - date_obj).days

        if days_diff > 10:
            return CheckResult(
                stock_code, CheckStatus.WARNING,
                f"数据滞后 {days_diff} 天 (最新：{latest_date})",
                {'latest_date': latest_date, 'days_behind': days_diff}
            )

        check_start = self._get_previous_date(str(latest_date), self.check_days)
        local_df = self.db.get_stock_data(
            stock_code,
            start_date=int(check_start),
            end_date=latest_date,
            chronological=False
        )

        if local_df is None or len(local_df) == 0:
            return CheckResult(stock_code, CheckStatus.FAIL, "本地数据为空")

        bs_start = self._date_to_standard_format(check_start)
        bs_end = self._date_to_standard_format(str(latest_date))
        bs_data = self.fetch_external_data(stock_code, bs_start, bs_end)

        if bs_data is None or len(bs_data) == 0:
            return CheckResult(
                stock_code, CheckStatus.SKIP,
                "无法获取外部数据（可能停牌）",
                {'latest_date': latest_date}
            )

        local_dates = set(str(d) for d in local_df['date'].values)
        bs_dates = set(str(d) for d in bs_data['date'].values)
        missing_dates = bs_dates - local_dates

        if missing_dates:
            return CheckResult(
                stock_code, CheckStatus.FAIL,
                f"缺失 {len(missing_dates)} 个交易日",
                {'missing_dates': sorted(list(missing_dates))}
            )

        merged = pd.merge(
            local_df.assign(date_str=local_df['date'].astype(str)),
            bs_data.assign(date_str=bs_data['date'].astype(str)),
            on='date_str', suffixes=('', '_bs')
        )

        if len(merged) > 0:
            open_bad = (merged['open'] - merged['open_bs']).abs() > 0.01
            if open_bad.any():
                r = merged.loc[open_bad.idxmax()]
                return CheckResult(
                    stock_code, CheckStatus.FAIL,
                    f"开盘价不匹配 ({r['date_str']}): 本地={r['open']}, 外部={r['open_bs']}",
                    {'date': r['date_str'], 'field': 'open'}
                )

            close_bad = (merged['close'] - merged['close_bs']).abs() > 0.01
            if close_bad.any():
                r = merged.loc[close_bad.idxmax()]
                return CheckResult(
                    stock_code, CheckStatus.FAIL,
                    f"收盘价不匹配 ({r['date_str']}): 本地={r['close']}, 外部={r['close_bs']}",
                    {'date': r['date_str'], 'field': 'close'}
                )

            amount_pct = (merged['amount'] - merged['amount_bs']).abs() / merged['amount_bs'].clip(lower=1)
            amount_bad = amount_pct > 0.05
            if amount_bad.any():
                r = merged.loc[amount_bad.idxmax()]
                return CheckResult(
                    stock_code, CheckStatus.WARNING,
                    f"成交额差异较大 ({r['date_str']}): 本地={r['amount']}, 外部={r['amount_bs']}",
                    {'date': r['date_str'], 'field': 'amount'}
                )

        return CheckResult(
            stock_code, CheckStatus.PASS,
            f"数据完整且准确 (最新：{latest_date}, 检查 {len(bs_data)} 个交易日)",
            {'latest_date': latest_date, 'checked_days': len(bs_data)}
        )

    def check_ohlc_logic(self, stock_code: str) -> CheckResult:
        """检查 OHLC 逻辑（向量化）"""
        df = self.db.get_stock_data(
            stock_code,
            columns=['date', 'open', 'high', 'low', 'close'],
            chronological=False
        )
        if len(df) == 0:
            return CheckResult(stock_code, CheckStatus.FAIL, "数据为空")

        df = df.head(self.check_days)

        bad = df[df['high'] < df['low']]
        if not bad.empty:
            r = bad.iloc[0]
            return CheckResult(stock_code, CheckStatus.FAIL,
                               f"最高价 < 最低价 ({r['date']})",
                               {'date': r['date'], 'high': r['high'], 'low': r['low']})

        bad = df[(df['open'] < df['low']) | (df['open'] > df['high'])]
        if not bad.empty:
            r = bad.iloc[0]
            return CheckResult(stock_code, CheckStatus.FAIL,
                               f"开盘价超出范围 ({r['date']})",
                               {'date': r['date'], 'open': r['open'], 'high': r['high'], 'low': r['low']})

        bad = df[(df['close'] < df['low']) | (df['close'] > df['high'])]
        if not bad.empty:
            r = bad.iloc[0]
            return CheckResult(stock_code, CheckStatus.FAIL,
                               f"收盘价超出范围 ({r['date']})",
                               {'date': r['date'], 'close': r['close'], 'high': r['high'], 'low': r['low']})

        return CheckResult(stock_code, CheckStatus.PASS,
                           f"OHLC 逻辑正确 (检查 {len(df)} 天)")

    def check_price_changes(self, stock_code: str) -> CheckResult:
        """检查涨跌幅是否合理（向量化）"""
        df = self.db.get_stock_data(
            stock_code,
            columns=['date', 'close'],
            chronological=False
        )
        if len(df) < 2:
            return CheckResult(stock_code, CheckStatus.SKIP, "数据不足")

        df = df.head(self.check_days).reset_index(drop=True)
        prev_close = df['close'].shift(1)
        valid = prev_close > 0
        change_pct = (df['close'] - prev_close) / prev_close
        abnormal = valid & (change_pct.abs() > 0.20)

        if abnormal.any():
            idx = abnormal.idxmax()
            return CheckResult(stock_code, CheckStatus.WARNING,
                               f"涨跌幅异常 ({df.iloc[idx]['date']}): {change_pct.iloc[idx] * 100:.2f}%",
                               {'date': df.iloc[idx]['date'], 'change': change_pct.iloc[idx]})

        return CheckResult(stock_code, CheckStatus.PASS,
                           f"涨跌幅正常 (检查 {len(df) - 1} 次)")

    def repair_stock_data(self, stock_code: str, result: CheckResult) -> bool:
        """尝试修复检测到的数据错误"""
        msg = result.message

        # 情况1：缺失交易日
        if "缺失" in msg and result.details and 'missing_dates' in result.details:
            missing_dates = result.details['missing_dates']
            print(f"  → 修复：补拉 {len(missing_dates)} 个缺失交易日")
            start_bs = f"{missing_dates[0][:4]}-{missing_dates[0][4:6]}-{missing_dates[0][6:]}"
            end_bs = f"{missing_dates[-1][:4]}-{missing_dates[-1][4:6]}-{missing_dates[-1][6:]}"
            patch_df = self.fetch_external_data(stock_code, start_bs, end_bs)
            if patch_df is None or len(patch_df) == 0:
                print(f"  ✗ 无法获取补丁数据")
                return False
            try:
                # 过滤已有日期
                existing_dates = set(str(d) for d in self.db.get_stock_data(
                    stock_code, columns=['date'], chronological=False)['date'].values)
                patch_df = patch_df[~patch_df['date'].astype(str).isin(existing_dates)]
                if len(patch_df) == 0:
                    print(f"  ⚠ 补丁数据已存在")
                    return True
                self.db.upsert_daily_data(stock_code, patch_df)
                print(f"  ✓ 补入 {len(patch_df)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        # 情况2：价格不匹配 / OHLC 逻辑错误 → 全量重拉（INSERT OR REPLACE 覆盖，无需先删）
        if any(k in msg for k in ["不匹配", "OHLC", "超出范围", "无法读取", "最高价"]):
            print(f"  → 修复：重新全量拉取 {stock_code}")
            df = self.fetch_external_data(
                stock_code, "2010-01-01",
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            if df is None or len(df) == 0:
                print(f"  ✗ 全量拉取失败")
                return False
            try:
                self.db.upsert_daily_data(stock_code, df)
                compute_features_for_stock(self.db, stock_code)
                print(f"  ✓ 全量写入 {len(df)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        # 情况3：数据滞后
        if "滞后" in msg and result.details and 'latest_date' in result.details:
            latest = result.details['latest_date']
            start_date = f"{str(latest)[:4]}-{str(latest)[4:6]}-{str(latest)[6:]}"
            next_day = (datetime.datetime.strptime(start_date, "%Y-%m-%d")
                        + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  → 修复：增量补更 {stock_code}（从 {next_day} 起）")
            patch_df = self.fetch_external_data(
                stock_code, next_day,
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            if patch_df is None or len(patch_df) == 0:
                print(f"  ⚠ 无新数据可补")
                return True
            try:
                self.db.upsert_daily_data(stock_code, patch_df)
                compute_features_for_stock(self.db, stock_code)
                new_latest = self.db.get_latest_date(stock_code)
                print(f"  ✓ 补入 {len(patch_df)} 条，最新：{new_latest}")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        print(f"  ⚠ 无对应修复策略：{msg}")
        return False

    def run_full_check(self, stock_codes: List[str] = None,
                       pool_type: str = 'all',
                       verbose: bool = False) -> List[CheckResult]:
        """运行完整的数据检查，发现错误时自动修复"""
        print(f"检查范围：最近 {self.check_days} 天")
        print(f"检查项目：数据完整性、OHLC 逻辑、涨跌幅合理性")

        if not self.login():
            print("无法登录数据源，退出检查")
            return []

        try:
            if stock_codes is None:
                stock_codes = self.db.get_pool_stocks(pool_type)
            print(f"\n检查股票：{len(stock_codes)} 只 (pool={pool_type})")

            if self.enable_backup:
                self.db.backup_database()

            results = []
            repair_success = 0
            repair_fail = 0

            for i, stock_code in enumerate(stock_codes, 1):
                if verbose:
                    print(f"\n[{i}/{len(stock_codes)}] 检查 {stock_code}...")
                else:
                    print(f"[{i}/{len(stock_codes)}] 检查 {stock_code}...", end=" ")

                integrity_result = self.check_data_integrity(stock_code, verbose)
                results.append(integrity_result)
                self._update_stats(integrity_result.status)

                if not verbose:
                    print(f"{integrity_result.status.value} - {integrity_result.message}")

                if integrity_result.status in (CheckStatus.FAIL, CheckStatus.WARNING):
                    ok = self.repair_stock_data(stock_code, integrity_result)
                    if ok:
                        repair_success += 1
                    else:
                        repair_fail += 1

                if integrity_result.status == CheckStatus.PASS:
                    ohlc_result = self.check_ohlc_logic(stock_code)
                    if ohlc_result.status != CheckStatus.PASS:
                        results.append(ohlc_result)
                        if not verbose:
                            print(f"  {ohlc_result.status.value} - {ohlc_result.message}")
                        self.repair_stock_data(stock_code, ohlc_result)

                    price_result = self.check_price_changes(stock_code)
                    if price_result.status != CheckStatus.PASS:
                        results.append(price_result)
                        if not verbose:
                            print(f"  {price_result.status.value} - {price_result.message}")

                if verbose and i % 50 == 0:
                    time.sleep(0.5)

            self._print_summary(results, repair_success, repair_fail)
            return results

        finally:
            self.logout()

    def _update_stats(self, status: CheckStatus):
        self.stats['total'] += 1
        status_map = {
            CheckStatus.PASS: 'pass',
            CheckStatus.FAIL: 'fail',
            CheckStatus.WARNING: 'warning',
            CheckStatus.SKIP: 'skip'
        }
        key = status_map.get(status)
        if key:
            self.stats[key] += 1

    def _print_summary(self, results: List[CheckResult],
                       repair_success: int = 0, repair_fail: int = 0):
        print("*" * 32 + " 检查摘要 " + "*" * 32)
        total = self.stats['total']
        print(f"总检查数：{total}")
        if total:
            print(f"✓ 通过：{self.stats['pass']} ({self.stats['pass'] / total * 100:.1f}%)")
        print(f"✗ 失败：{self.stats['fail']}")
        print(f"⚠ 警告：{self.stats['warning']}")
        print(f"○ 跳过：{self.stats['skip']}")

        failed_results = [r for r in results if r.status == CheckStatus.FAIL]
        warning_results = [r for r in results if r.status == CheckStatus.WARNING]

        if failed_results:
            print(f"\n失败股票 ({len(failed_results)}):")
            for r in failed_results[:10]:
                print(f"  - {r.stock_code}: {r.message}")
            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 只")

        if warning_results:
            print(f"\n警告股票 ({len(warning_results)}):")
            for r in warning_results[:10]:
                print(f"  - {r.stock_code}: {r.message}")
            if len(warning_results) > 10:
                print(f"  ... 还有 {len(warning_results) - 10} 只")

        if repair_success + repair_fail > 0:
            print(f"\n修复统计：成功 {repair_success}，失败 {repair_fail}")

    @staticmethod
    def _format_stock_code(stock_code: str) -> Optional[str]:
        if stock_code.startswith('6') or stock_code.startswith('9'):
            return f"sh.{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            return f"sz.{stock_code}"
        return None

    @staticmethod
    def _date_to_standard_format(date_str: str) -> str:
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    @staticmethod
    def _get_previous_date(date_str: str, days: int) -> str:
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
        return (date_obj - datetime.timedelta(days=days)).strftime("%Y%m%d")


class AKShareDataChecker(DataChecker):
    """基于 AKShare 的数据质量检查器"""

    def __init__(self, db: DatabaseManager, check_days: int = 100, backup: bool = True):
        super().__init__(db, check_days, backup)
        import akshare as ak
        self.ak = ak

    def login(self) -> bool:
        return True

    def logout(self):
        pass

    def fetch_external_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            start_fmt = start_date.replace('-', '')
            end_fmt = end_date.replace('-', '')
            df = self.ak.stock_zh_a_hist(
                symbol=stock_code, period="daily",
                start_date=start_fmt, end_date=end_fmt, adjust=""
            )
            if df is None or len(df) == 0:
                return None
            return normalize_stock_df(df, source='akshare', volume_scale_factor=100.0)
        except Exception as e:
            print(f"AKShare 获取数据失败：{e}")
            return None

    def _format_stock_code(self, stock_code: str):
        return stock_code


def create_checker(db: DatabaseManager, data_source: str = 'akshare',
                   check_days: int = 100, backup: bool = True):
    """工厂函数：根据数据源创建对应的检查器"""
    if data_source == 'akshare':
        return AKShareDataChecker(db, check_days, backup)
    else:
        return DataChecker(db, check_days, backup)

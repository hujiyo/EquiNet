"""
股票数据更新模块

支持 Baostock 和 AKShare 两种数据源。

更新模式：
- incremental (默认): 增量更新全量池(all)中已有股票
- full: 从数据源获取全部 A 股列表，全量拉取
- train: 增量更新训练池(selected)中的股票
"""

import time
import sys
import os
import json
import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Tuple

from .database import DatabaseManager
from .features import compute_features_for_stock
from .utils import normalize_stock_df

# 进度持久化文件路径
_PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backup', 'update_progress.json')


def _load_progress() -> dict:
    """加载进度文件（如果存在）"""
    if os.path.exists(_PROGRESS_FILE):
        try:
            with open(_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_progress(progress: dict):
    """保存进度到文件"""
    os.makedirs(os.path.dirname(_PROGRESS_FILE), exist_ok=True)
    with open(_PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False)


def _clear_progress():
    """删除进度文件"""
    if os.path.exists(_PROGRESS_FILE):
        os.remove(_PROGRESS_FILE)


class StockDataUpdater:
    """基于 Baostock 的数据更新器"""

    _rate_limit_interval = 5   # 每N只股票暂停1秒，子类可覆盖
    _query_timeout = 60        # 单次 Baostock 查询超时（秒）

    def __init__(self, db: DatabaseManager, backup: bool = True):
        self.db = db
        self.enable_backup = backup
        self.login_success = False

    def login(self) -> bool:
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == '0':
                self.login_success = True
                self.bs = bs
                # 给底层 socket 设置超时，防止服务端限流时无限阻塞
                import baostock.common.context as bs_ctx
                if hasattr(bs_ctx, 'default_socket') and bs_ctx.default_socket is not None:
                    bs_ctx.default_socket.settimeout(self._query_timeout)
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

    @staticmethod
    def _is_a_share(exchange: str, code: str) -> bool:
        """判断是否为 A 股股票代码（排除指数、ETF、债券、基金等）"""
        if len(code) != 6 or not code.isdigit():
            return False
        if exchange == 'sh' and code[0] == '6':       # 上交所：主板(60x) + 科创板(688)
            return True
        if exchange == 'sz' and code[0] in ('0', '3'): # 深交所：主板(00x) + 创业板(30x)
            return not code.startswith('39')            # 排除深证系列指数(399xxx)
        return False

    def get_all_a_shares(self) -> List[Tuple[str, str]]:
        """获取所有 A 股列表 [(code, name), ...]"""
        stock_list = []
        try:
            rs = self.bs.query_all_stock(day=datetime.datetime.now().strftime("%Y-%m-%d"))
            while rs.error_code == '0' and rs.next():
                info = rs.get_row_data()
                code = info[0]        # e.g. 'sh.600000'
                name = info[1]
                parts = code.split('.')
                if len(parts) == 2 and self._is_a_share(parts[0], parts[1]):
                    stock_list.append((parts[1], name))
        except Exception as e:
            print(f"获取股票列表失败：{e}")
        return stock_list

    def _query_and_parse(self, code_with_prefix: str, query_start: str, end_date: str) -> Optional[pd.DataFrame]:
        """实际执行 Baostock 查询并解析（在子线程中运行）"""
        rs = self.bs.query_history_k_data_plus(
            code_with_prefix,
            "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,"
            "pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=query_start,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )

        if rs.error_code != '0':
            return None

        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        return normalize_stock_df(df, source='baostock')

    def fetch_stock_data(self, stock_code: str, start_date: str = None) -> Optional[pd.DataFrame]:
        """获取单只股票的 K 线数据（带超时保护）"""
        try:
            code_with_prefix = self._format_stock_code(stock_code)
            if code_with_prefix is None:
                return None

            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            query_start = start_date if start_date else "2010-01-01"

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._query_and_parse, code_with_prefix, query_start, end_date)
                return future.result(timeout=self._query_timeout)

        except FuturesTimeoutError:
            print(f"超时({self._query_timeout}s)", end=" ")
            return None
        except Exception as e:
            print(f"✗ {stock_code} 数据处理异常：{e}")
            return None

    @staticmethod
    def _latest_possible_trading_date() -> int:
        """
        推算当前时刻最近一个「已收盘的交易日」日期（YYYYMMDD 整数）。

        规则：
        1. A 股 15:00 收盘，收盘前当天数据不完整，应取前一交易日。
        2. 周末（周六/周日）自动回退到周五。
        3. 法定节假日无法穷举，保守处理：只跳过确定的周末，
           其余情况不跳过，留给服务端判断。
        """
        now = datetime.datetime.now()
        # 服务器通常 18:00 后才有当日数据，此前视为未更新
        if now.hour < 18:
            now -= datetime.timedelta(days=1)

        weekday = now.weekday()  # 0=Mon ... 6=Sun
        if weekday == 5:         # 周六 → 回退到周五
            now -= datetime.timedelta(days=1)
        elif weekday == 6:       # 周日 → 回退到周五
            now -= datetime.timedelta(days=2)

        return int(now.strftime("%Y%m%d"))

    def update_single_stock(self, stock_code: str, incremental: bool = True,
                            pool_type: str = 'all') -> bool:
        """更新单只股票数据到数据库"""
        if incremental:
            last_date = self.db.get_latest_date(stock_code)
            if last_date:
                # 本地预判：如果该票最新日期已经达到最近可能交易日，直接跳过
                cutoff = self._latest_possible_trading_date()
                if last_date >= cutoff:
                    print(f"✓ {stock_code} 已最新 (最新：{last_date})")
                    return True

                # 如果最后数据距今超过 365 天，视为已退市或长期停牌，跳过更新
                last_dt = datetime.datetime.strptime(str(last_date), "%Y%m%d")
                days_gap = (datetime.datetime.now() - last_dt).days
                if days_gap > 365:
                    print(f"⊘ {stock_code} 长期无数据（最后：{last_date}，{days_gap}天前），跳过")
                    return True

                start_str = f"{str(last_date)[:4]}-{str(last_date)[4:6]}-{str(last_date)[6:]}"
                next_date = (datetime.datetime.strptime(start_str, "%Y-%m-%d")
                            + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                df = self.fetch_stock_data(stock_code, start_date=next_date)
                if df is None:
                    print(f"✗ {stock_code} 数据拉取失败")
                    return False
                if df.empty:
                    print(f"✓ {stock_code} 已最新 (最新：{last_date})")
                    return True
                with self.db.transaction():
                    self.db.upsert_daily_data(stock_code, df, auto_commit=False)
                    compute_features_for_stock(self.db, stock_code, auto_commit=False)
                new_latest = self.db.get_latest_date(stock_code)
                print(f"✓ {stock_code} 增量更新：{last_date} → {new_latest} (新增 {len(df)} 条)")
                return True
            else:
                df = self.fetch_stock_data(stock_code)
                if df is None or df.empty:
                    print(f"✗ {stock_code} 获取失败")
                    return False
                with self.db.transaction():
                    self.db.upsert_daily_data(stock_code, df, auto_commit=False)
                    self.db.add_to_pool([stock_code], pool_type, auto_commit=False)
                    compute_features_for_stock(self.db, stock_code, auto_commit=False)
                print(f"✓ {stock_code} 全量保存：{len(df)} 条")
                return True
        else:
            df = self.fetch_stock_data(stock_code)
            if df is None or df.empty:
                print(f"✗ {stock_code} 全量更新失败")
                return False
            with self.db.transaction():
                self.db.upsert_daily_data(stock_code, df, auto_commit=False)
                self.db.add_to_pool([stock_code], pool_type, auto_commit=False)
                compute_features_for_stock(self.db, stock_code, auto_commit=False)
            print(f"✓ {stock_code} 全量更新：{len(df)} 条")
            return True

    def update_all_stocks(self, mode: str = 'incremental', stock_codes: List[str] = None):
        """批量更新所有股票数据，支持中断续传"""
        pool_type = 'selected' if mode == 'train' else 'all'

        if not self.login():
            print("无法登录数据源，退出更新")
            return

        # 检查是否有未完成的进度
        progress = _load_progress()
        completed_codes = set()
        resume_failed = []

        if progress and not stock_codes:
            prev_mode = progress.get('mode', '')
            prev_completed = progress.get('completed', [])
            prev_failed = progress.get('failed', [])
            print(f"\n发现未完成的 {prev_mode} 更新进度：已完成 {len(prev_completed)} 只，失败 {len(prev_failed)} 只")
            resume = input("是否从断点继续？(Y/n): ").strip().lower()
            if resume != 'n':
                completed_codes = set(prev_completed)
                resume_failed = prev_failed
                print(f"将从断点继续，跳过 {len(completed_codes)} 只已完成股票")
            else:
                _clear_progress()

        try:
            if stock_codes:
                codes_to_update = stock_codes
                print(f"\n更新指定股票：{len(stock_codes)} 只")
            elif resume_failed:
                codes_to_update = resume_failed
                print(f"\n重试失败股票：{len(codes_to_update)} 只")
            elif mode == 'full':
                print("\n全量更新模式：获取所有 A 股数据")
                all_stocks = self.get_all_a_shares()
                codes_to_update = [code for code, _ in all_stocks]
                print(f"获取到 {len(codes_to_update)} 只股票")
            else:
                codes_to_update = self.db.get_pool_stocks(pool_type)
                print(f"\n增量更新模式：更新 {len(codes_to_update)} 只已有股票 (pool={pool_type})")

            if self.enable_backup:
                self.db.backup_database()

            success_count = 0
            failed_stocks = []
            consecutive_fail = 0
            incremental = (mode != 'full')
            total = len(codes_to_update)

            try:
                for i, stock_code in enumerate(codes_to_update, 1):
                    if stock_code in completed_codes:
                        continue
                    print(f"[{i}/{total}] 更新 {stock_code}...", end=" ")
                    try:
                        if self.update_single_stock(stock_code, incremental, pool_type):
                            success_count += 1
                            consecutive_fail = 0
                            completed_codes.add(stock_code)
                        else:
                            failed_stocks.append(stock_code)
                            consecutive_fail += 1
                    except Exception as e:
                        print(f"✗ 异常：{e}")
                        failed_stocks.append(stock_code)
                        consecutive_fail += 1

                    # 连续失败 3 次 → 重连
                    if consecutive_fail >= 3:
                        print("\n  ⚠ 连续失败，尝试重新登录...")
                        self.logout()
                        time.sleep(3)
                        if self.login():
                            consecutive_fail = 0
                        else:
                            print("  ✗ 重连失败，终止更新")
                            break

                    if i % self._rate_limit_interval == 0:
                        time.sleep(1)

                    # 每 50 只股票保存一次进度
                    if i % 50 == 0:
                        _save_progress({
                            'mode': mode,
                            'completed': list(completed_codes),
                            'failed': failed_stocks,
                            'timestamp': datetime.datetime.now().isoformat()
                        })

            except KeyboardInterrupt:
                _save_progress({
                    'mode': mode,
                    'completed': list(completed_codes),
                    'failed': failed_stocks,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                print(f"\n\n⚠ 中断！已完成 {len(completed_codes)} 只，失败 {len(failed_stocks)} 只")
                print("进度已保存，重新运行可选择断点续传。")
                return

            _clear_progress()

            print("*" * 32 + " 更新完成统计 " + "*" * 32)
            print(f"成功：{success_count}/{total}")
            print(f"失败：{len(failed_stocks)}")

            if failed_stocks:
                failed_str = ', '.join(failed_stocks[:20])
                if len(failed_stocks) > 20:
                    failed_str += '...'
                print(f"\n失败的股票：{failed_str}")

        finally:
            self.logout()

    @staticmethod
    def _format_stock_code(stock_code: str) -> Optional[str]:
        if stock_code.startswith('6') or stock_code.startswith('9'):
            return f"sh.{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            return f"sz.{stock_code}"
        else:
            return None


class AKShareDataUpdater(StockDataUpdater):
    """基于 AKShare 的数据更新器"""

    _rate_limit_interval = 5  # AKShare 限流更严格

    def __init__(self, db: DatabaseManager, backup: bool = True):
        super().__init__(db, backup)
        import akshare as ak
        self.ak = ak

    def login(self) -> bool:
        return True

    def logout(self):
        pass

    def get_all_a_shares(self) -> List[Tuple[str, str]]:
        try:
            df = self.ak.stock_zh_a_spot_em()
            stock_list = []
            for _, row in df.iterrows():
                code = str(row['代码'])
                name = str(row['名称'])
                if code.startswith(('6', '0', '3')):
                    stock_list.append((code, name))
            return stock_list
        except Exception as e:
            print(f"✗ AKShare 获取股票列表失败：{e}")
            return []

    def fetch_stock_data(self, stock_code: str, start_date: str = None) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            query_start = start_date.replace('-', '') if start_date else "20100101"

            df = self.ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=query_start,
                end_date=end_date,
                adjust=""
            )

            if df is None or len(df) == 0:
                return None

            return normalize_stock_df(df, source='akshare', volume_scale_factor=100.0)

        except Exception as e:
            print(f"✗ {stock_code} AKShare 获取失败：{e}")
            return None


def create_updater(db: DatabaseManager, data_source: str = 'akshare', backup: bool = True):
    """工厂函数：根据数据源创建对应的更新器"""
    if data_source == 'akshare':
        return AKShareDataUpdater(db, backup)
    else:
        return StockDataUpdater(db, backup)

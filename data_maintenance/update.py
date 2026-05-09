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
import datetime
import pandas as pd
from typing import List, Optional, Tuple

from .database import DatabaseManager


class StockDataUpdater:
    """基于 Baostock 的数据更新器"""

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

    def get_all_a_shares(self) -> List[Tuple[str, str]]:
        """获取所有 A 股列表 [(code, name), ...]"""
        stock_list = []
        try:
            rs = self.bs.query_all_stock(day=datetime.datetime.now().strftime("%Y-%m-%d"))
            while rs.error_code == '0' and rs.next():
                info = rs.get_row_data()
                code = info[0]
                name = info[1]
                if code.startswith('sh') or code.startswith('sz'):
                    stock_code = code.split('.')[1]
                    stock_list.append((stock_code, name))
        except Exception as e:
            print(f"获取股票列表失败：{e}")
        return stock_list

    def fetch_stock_data(self, stock_code: str, start_date: str = None) -> Optional[pd.DataFrame]:
        """获取单只股票的 K 线数据（不复权）"""
        columns = ['date', 'open', 'high', 'low', 'close', 'amount', 'volume', 'exchange', 'vwap']
        try:
            code_with_prefix = self._format_stock_code(stock_code)
            if code_with_prefix is None:
                return None

            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            query_start = start_date if start_date else "2010-01-01"

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
                return pd.DataFrame(columns=columns)

            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = df['date'].str.replace('-', '')
            df = df[df['date'] != '']
            if len(df) == 0:
                return pd.DataFrame(columns=columns)
            df['date'] = df['date'].astype(int)

            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
            df['vwap'] = df['amount'] / df['volume'].replace(0, float('nan'))
            df['vwap'] = df['vwap'].fillna(df['close'])
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            if len(df) == 0:
                return None

            if 'turn' in df.columns:
                df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
                df['exchange'] = df['turn'].fillna(0.0).astype(float)
            else:
                df['exchange'] = 0.0

            df = df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume', 'exchange', 'vwap']]
            return df

        except Exception as e:
            print(f"✗ {stock_code} 数据处理异常：{e}")
            return None

    def update_single_stock(self, stock_code: str, incremental: bool = True,
                            pool_type: str = 'all') -> bool:
        """更新单只股票数据到数据库"""
        if incremental:
            last_date = self.db.get_latest_date(stock_code)
            if last_date:
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
                self.db.upsert_daily_data(stock_code, df)
                new_latest = self.db.get_latest_date(stock_code)
                print(f"✓ {stock_code} 增量更新：{last_date} → {new_latest} (新增 {len(df)} 条)")
                return True
            else:
                df = self.fetch_stock_data(stock_code)
                if df is None or df.empty:
                    print(f"✗ {stock_code} 获取失败")
                    return False
                self.db.upsert_daily_data(stock_code, df)
                self.db.add_to_pool([stock_code], pool_type)
                print(f"✓ {stock_code} 全量保存：{len(df)} 条")
                return True
        else:
            df = self.fetch_stock_data(stock_code)
            if df is None or df.empty:
                print(f"✗ {stock_code} 全量更新失败")
                return False
            self.db.upsert_daily_data(stock_code, df)
            self.db.add_to_pool([stock_code], pool_type)
            print(f"✓ {stock_code} 全量更新：{len(df)} 条")
            return True

    def update_all_stocks(self, mode: str = 'incremental', stock_codes: List[str] = None):
        """批量更新所有股票数据"""
        pool_type = 'selected' if mode == 'train' else 'all'

        if not self.login():
            print("无法登录数据源，退出更新")
            return

        try:
            if stock_codes:
                codes_to_update = stock_codes
                print(f"\n更新指定股票：{len(stock_codes)} 只")
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
            incremental = (mode != 'full')

            for i, stock_code in enumerate(codes_to_update, 1):
                print(f"[{i}/{len(codes_to_update)}] 更新 {stock_code}...", end=" ")
                try:
                    if self.update_single_stock(stock_code, incremental, pool_type):
                        success_count += 1
                    else:
                        failed_stocks.append(stock_code)
                except Exception as e:
                    print(f"✗ 异常：{e}")
                    failed_stocks.append(stock_code)

                if i % 50 == 0:
                    time.sleep(1)

            print("*" * 32 + " 更新完成统计 " + "*" * 32)
            print(f"成功：{success_count}/{len(codes_to_update)}")
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

            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close',
                '成交量': 'volume', '成交额': 'amount', '换手率': 'exchange'
            })

            df = df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume', 'exchange']]
            df['date'] = df['date'].astype(str).str.replace('-', '').astype(int)

            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0) * 100
            df['exchange'] = pd.to_numeric(df['exchange'], errors='coerce').fillna(0.0)
            df['vwap'] = df['amount'] / df['volume'].replace(0, float('nan'))
            df['vwap'] = df['vwap'].fillna(df['close'])
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            if len(df) == 0:
                return None

            df = df[['date', 'open', 'high', 'low', 'close', 'amount', 'volume', 'exchange', 'vwap']]
            return df

        except Exception as e:
            print(f"✗ {stock_code} AKShare 获取失败：{e}")
            return None

    def update_all_stocks(self, mode: str = 'incremental', stock_codes: List[str] = None):
        """覆写以增加 AKShare 的请求间隔"""
        pool_type = 'selected' if mode == 'train' else 'all'

        if stock_codes:
            codes_to_update = stock_codes
            print(f"\n更新指定股票：{len(stock_codes)} 只")
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
        incremental = (mode != 'full')

        for i, stock_code in enumerate(codes_to_update, 1):
            print(f"[{i}/{len(codes_to_update)}] 更新 {stock_code}...", end=" ")
            try:
                if self.update_single_stock(stock_code, incremental, pool_type):
                    success_count += 1
                else:
                    failed_stocks.append(stock_code)
            except Exception as e:
                print(f"✗ 异常：{e}")
                failed_stocks.append(stock_code)

            if i % 5 == 0:
                time.sleep(1)

        print("*" * 32 + " 更新完成统计 " + "*" * 32)
        print(f"成功：{success_count}/{len(codes_to_update)}")
        print(f"失败：{len(failed_stocks)}")

        if failed_stocks:
            failed_str = ', '.join(failed_stocks[:20])
            if len(failed_stocks) > 20:
                failed_str += '...'
            print(f"\n失败的股票：{failed_str}")


def create_updater(db: DatabaseManager, data_source: str = 'akshare', backup: bool = True):
    """工厂函数：根据数据源创建对应的更新器"""
    if data_source == 'akshare':
        return AKShareDataUpdater(db, backup)
    else:
        return StockDataUpdater(db, backup)

"""
股票数据更新脚本

使用 Baostock 获取 A 股历史行情数据，自动更新到最新交易日
支持增量更新、全量更新、训练数据更新三种模式

功能特性：
- 自动识别已有数据，支持增量更新
- 使用不复权数据（原始价格），确保历史数据不变
- 自动创建备份
- 支持指定股票或批量更新所有股票
- 详细的更新日志和错误处理

更新模式说明：
- incremental (默认)：增量更新 data_all/ 中已有股票的最新数据，适合日常维护
- full：从 Baostock 获取全部 A 股列表，重新拉取完整历史数据，适合初始化或添加新股
- train：增量更新 data/ 中已筛选的股票（由 data_select.py 生成），适合训练前快速刷新
  由于 data/ 中的股票数量远少于 data_all/，train 模式速度很快，适合频繁使用

使用方法:
python data_update.py                              # 增量更新 data_all/ 全量股票池 (默认)
python data_update.py --mode full                  # 全量更新 data_all/ 所有 A 股
python data_update.py --mode train                 # 增量更新 data/ 训练数据（快速）
python data_update.py --stocks 000001 600000       # 更新指定股票
python data_update.py --no-backup                  # 禁用备份
"""

import time
import datetime
import pandas as pd
import baostock as bs
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import argparse

class StockDataUpdater:
    def __init__(self, data_dir: str, backup: bool = True):
        """
        Args:
            data_dir: 数据存储目录
            backup: 是否启用备份（默认启用）
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir.parent / "data_backup"
        self.enable_backup = backup
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.login_success = False

    def login_baostock(self) -> bool:
        try:
            lg = bs.login()
            if lg.error_code == '0':
                self.login_success = True
                return True
            else:
                print(f"✗ Baostock 登录失败：{lg.error_msg}")
                return False
        except Exception as e:
            print(f"✗ Baostock 登录异常：{e}")
            return False
    
    def logout_baostock(self):
        if self.login_success:
            bs.logout()
            self.login_success = False
    
    def get_existing_stocks(self) -> set:
        """获取已存在的股票代码集合"""
        existing_stocks = set()
        for file in self.data_dir.glob("*.csv"):
            stock_code = file.stem
            existing_stocks.add(stock_code)
        return existing_stocks
    
    def get_all_a_shares(self) -> List[Tuple[str, str]]:
        """
        获取所有 A 股股票列表
        
        Returns:
            [(股票代码，股票名称), ...]
        """
        stock_list = []
        try:
            rs = bs.query_all_stock(day=datetime.datetime.now().strftime("%Y-%m-%d"))
            while rs.error_code == '0' and rs.next():
                stock_info = rs.get_row_data()
                code = stock_info[0]
                code_name = stock_info[1]
                
                if code.startswith('sh') or code.startswith('sz'):
                    stock_code = code.split('.')[1]
                    stock_list.append((stock_code, code_name))
        except Exception as e:
            print(f"获取股票列表失败：{e}")
        
        return stock_list
    
    def get_last_date_in_file(self, stock_code: str) -> Optional[str]:
        """
        获取文件中最新的日期
        Args:
            stock_code: 股票代码
        Returns:
            最新日期字符串 (YYYYMMDD 格式) 或 None
        """
        file_path = self.data_dir / f"{stock_code}.csv"
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                latest_date = df.iloc[0]['date']
                if isinstance(latest_date, (int, float)):
                    latest_date = str(int(latest_date))
                else:
                    latest_date = str(latest_date)
                if len(latest_date) == 8:
                    return latest_date
                return None
            return None
        except Exception as e:
            print(f"读取文件 {stock_code}.csv 失败：{e}")
            return None
    
    def backup_data(self):
        """备份现有数据"""
        if not self.enable_backup:
            return
        
        try:
            if self.data_dir.exists():
                backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / backup_timestamp
                backup_path.mkdir(parents=True, exist_ok=True)
                
                for csv_file in self.data_dir.glob("*.csv"):
                    shutil.copy2(csv_file, backup_path / csv_file.name)
                
                print(f"✓ 数据已备份到：{backup_path}")
        except Exception as e:
            print(f"✗ 备份失败：{e}")
    
    def fetch_stock_data(self, stock_code: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取单只股票的 K 线数据（不复权）
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期 (YYYY-MM-DD 格式)，None 表示获取所有数据
        Returns:
            DataFrame 或 None
        """
        columns = ['date', 'open', 'high', 'low', 'close', 'amount', 'exchange']

        try:
            code_with_prefix = self._format_stock_code(stock_code)
            if code_with_prefix is None:
                return None
            
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            if start_date:
                query_start = start_date
            else:
                query_start = "2010-01-01"
            
            rs = bs.query_history_k_data_plus(
                code_with_prefix,
                "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,"
                "pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date=query_start,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                print(f"✗ {stock_code} 数据获取失败：{rs.error_msg}")
                return None
            
            data_list = []
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                data_list.append(row)
            
            if not data_list:
                return pd.DataFrame(columns=columns)
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            df['date'] = df['date'].str.replace('-', '')
            df = df[df['date'] != '']
            if len(df) == 0:
                return pd.DataFrame(columns=columns)
            df['date'] = df['date'].astype(int)
            
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df['amount'] = (df['amount'] / 1000.0).fillna(0.0)
            else:
                print(f"⚠ {stock_code} 警告：Baostock 未返回 amount 字段，amount 将设为 0")
                df['amount'] = pd.Series([0.0] * len(df), dtype=float)
            
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) == 0:
                return None
            
            if 'turn' in df.columns:
                df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
                df['exchange'] = df['turn'].fillna(0.0).astype(float)
            else:
                df['exchange'] = 0.0
            
            df = df[['date', 'open', 'high', 'low', 'close', 'amount', 'exchange']]
            df = df.iloc[::-1].reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"✗ {stock_code} 数据处理异常：{e}")
            return None

    def _format_stock_code(self, stock_code: str) -> Optional[str]:
        """
        格式化股票代码为 Baostock 格式
        
        Args:
            stock_code: 原始股票代码
        Returns:
            格式化后的代码 (如 sh.600000) 或 None
        """
        if stock_code.startswith('6') or stock_code.startswith('9'):
            return f"sh.{stock_code}"
        elif stock_code.startswith('0') or stock_code.startswith('3'):
            return f"sz.{stock_code}"
        else:
            print(f"✗ 未知的股票代码格式：{stock_code}")
            return None
    
    def save_stock_data(self, stock_code: str, df: pd.DataFrame, incremental: bool = False, old_latest_date: Optional[str] = None) -> bool:
        """
        保存股票数据到 CSV 文件
        
        Args:
            stock_code: 股票代码
            df: 数据 DataFrame
            incremental: 是否为增量更新
            old_latest_date: 原文件中最新日期（仅增量更新时使用）
        Returns:
            是否保存成功
        """
        try:
            file_path = self.data_dir / f"{stock_code}.csv"
            
            if incremental and file_path.exists():
                old_df = pd.read_csv(file_path)
                
                new_dates = set(df['date'].astype(str))
                old_dates = set(old_df['date'].astype(str))
                
                existing_dates = new_dates & old_dates
                if existing_dates:
                    df = df[~df['date'].astype(str).isin(existing_dates)]
                
                if len(df) == 0:
                    return True
                
                combined_df = pd.concat([df, old_df], ignore_index=True)
                combined_df = combined_df.sort_values('date', ascending=False).reset_index(drop=True)
                combined_df.to_csv(file_path, index=False)
                
                new_latest = str(int(combined_df.iloc[0]['date']))
                if old_latest_date:
                    print(f"✓ {stock_code} 增量更新：{old_latest_date} → {new_latest} (新增 {len(df)} 条，总计 {len(combined_df)} 条)")
                else:
                    print(f"✓ {stock_code} 增量更新：新增 {len(df)} 条，总计 {len(combined_df)} 条 (最新：{new_latest})")
                return True
            else:
                df.to_csv(file_path, index=False)
                new_latest = str(int(df.iloc[0]['date'])) if len(df) > 0 else 'N/A'
                print(f"✓ {stock_code} 全量保存：{len(df)} 条 (最新：{new_latest})")
                return True
                
        except Exception as e:
            print(f"✗ {stock_code} 保存失败：{e}")
            return False
    
    def update_single_stock(self, stock_code: str, incremental: bool = True) -> bool:
        """
        更新单只股票数据
        
        Args:
            stock_code: 股票代码
            incremental: 是否增量更新
            
        Returns:
            是否更新成功
        """
        if incremental:
            last_date = self.get_last_date_in_file(stock_code)
            if last_date:
                start_date = self._convert_date_format(last_date)
                if start_date:
                    next_date = self._get_next_date(start_date)
                    df = self.fetch_stock_data(stock_code, start_date=next_date)
                    if df is None:
                        print(f"✗ {stock_code} 数据拉取失败，稍后重试")
                        return False
                    if df.empty:
                        print(f"✓ {stock_code} 已最新 (最新：{last_date})")
                        return True
                    return self.save_stock_data(stock_code, df, incremental=True, old_latest_date=last_date)
                else:
                    print(f"⚠ {stock_code} 日期格式转换失败，重新获取全量数据")
                    df = self.fetch_stock_data(stock_code)
                    if df is None or df.empty:
                        print(f"✗ {stock_code} 全量获取失败，稍后重试")
                        return False
                    return self.save_stock_data(stock_code, df, incremental=False)
            else:
                df = self.fetch_stock_data(stock_code)
                if df is None or df.empty:
                    print(f"✗ {stock_code} 获取失败，无法初始化数据")
                    return False
                return self.save_stock_data(stock_code, df, incremental=False)
        else:
            df = self.fetch_stock_data(stock_code)
            if df is None or df.empty:
                print(f"✗ {stock_code} 全量更新失败：未获取到数据")
                return False
            return self.save_stock_data(stock_code, df, incremental=False)
    
    def _convert_date_format(self, date_str: str) -> Optional[str]:
        """
        转换日期格式从 YYYYMMDD 到 YYYY-MM-DD
        
        Args:
            date_str: YYYYMMDD 格式的日期
        Returns:
            YYYY-MM-DD 格式的日期或 None
        """
        try:
            if len(date_str) == 8:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            return None
        except:
            return None
    
    def _get_next_date(self, date_str: str) -> str:
        """
        获取给定日期的下一天
        
        Args:
            date_str: YYYY-MM-DD 格式的日期
        Returns:
            下一天的日期 (YYYY-MM-DD)
        """
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            next_date = date_obj + datetime.timedelta(days=1)
            return next_date.strftime("%Y-%m-%d")
        except:
            return date_str
    
    def update_all_stocks(self, incremental: bool = True, stock_codes: Optional[List[str]] = None):
        """
        批量更新所有股票数据
        
        Args:
            incremental: 是否增量更新
            stock_codes: 指定要更新的股票代码列表，None 表示更新所有
        """
        if not self.login_baostock():
            print("无法登录 Baostock，退出更新")
            return
        
        try:
            if stock_codes:
                codes_to_update = stock_codes
                print(f"\n更新指定股票：{len(stock_codes)} 只")
            else:
                if incremental:
                    existing_stocks = self.get_existing_stocks()
                    codes_to_update = list(existing_stocks)
                    print(f"\n增量更新模式：更新 {len(codes_to_update)} 只已有股票")
                else:
                    print("\n全量更新模式：获取所有 A 股数据")
                    all_stocks = self.get_all_a_shares()
                    codes_to_update = [code for code, _ in all_stocks]
                    print(f"获取到 {len(codes_to_update)} 只股票")
            
            if self.enable_backup:
                self.backup_data()
            
            success_count = 0
            failed_stocks = []
            
            for i, stock_code in enumerate(codes_to_update, 1):
                print(f"[{i}/{len(codes_to_update)}] 更新 {stock_code}...", end=" ")
                
                try:
                    if self.update_single_stock(stock_code, incremental):
                        success_count += 1
                    else:
                        failed_stocks.append(stock_code)
                except Exception as e:
                    print(f"✗ 异常：{e}")
                    failed_stocks.append(stock_code)
                
                if i % 50 == 0:
                    time.sleep(1)
            
            print("*"*32 + "更新完成统计" + "*"*32)
            print(f"成功：{success_count}/{len(codes_to_update)}")
            print(f"失败：{len(failed_stocks)}")
            
            if failed_stocks:
                failed_str = ', '.join(failed_stocks[:20])
                if len(failed_stocks) > 20:
                    failed_str += '...'
                print(f"\n失败的股票：{failed_str}")
            
        finally:
            self.logout_baostock()


def main():
    parser = argparse.ArgumentParser(description='EquiNet股票数据更新工具')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据存储目录 (默认：由 --mode 决定)')
    parser.add_argument('--mode', type=str, choices=['incremental', 'full', 'train'], default='incremental',
                       help='更新模式：incremental(增量更新data_all) | full(全量更新data_all) | train(增量更新data)')
    parser.add_argument('--stocks', type=str, nargs='+',help='指定要更新的股票代码列表')
    parser.add_argument('--no-backup', action='store_true',help='禁用备份')
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = script_dir / data_dir
    else:
        if args.mode == 'train':
            data_dir = script_dir / 'data'
        else:
            data_dir = script_dir / 'data_all'

    print(f"数据目录：{data_dir}")
    print(f"更新模式：{args.mode}")

    updater = StockDataUpdater(str(data_dir), backup=not args.no_backup)

    incremental = (args.mode != 'full')
    updater.update_all_stocks(incremental=incremental, stock_codes=args.stocks)


if __name__ == "__main__":
    main()

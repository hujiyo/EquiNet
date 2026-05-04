"""
股票数据筛选脚本

从 data_all/ 全量股票池中筛选符合条件的股票，复制到 data/ 供模型训练使用

筛选条件：
- 主板股票：沪市(600/601/603/605) + 深市(000/001/002/003)
- 排除创业板(300xxx)、科创板(688xxx)、北交所(8xx/4xx)
- 排除ST/*ST股票（通过新浪股票名称判断）
- 排除已退市或长期停牌股票（最新数据超过30天）
- 市值筛选：仅保留市值在 [10亿, 200亿] 范围内的股票（使用流通市值）

使用方法：
python data_select.py                    # 使用默认配置筛选
python data_select.py --dry-run          # 仅打印筛选结果，不复制文件
"""

import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import datetime
import pandas as pd
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Dict

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))

from config import DataConfig

try:
    import akshare as ak
except ImportError:
    print("请先安装 akshare: pip install akshare")
    sys.exit(1)


class DataSelector:
    """从全量股票池中筛选符合条件的股票"""

    def __init__(self, data_all_dir: str, data_dir: str,
                 market_cap_min: float = None,
                 market_cap_max: float = None,
                 valid_prefixes: list = None):
        self.data_all_dir = Path(data_all_dir)
        self.data_dir = Path(data_dir)
        self.market_cap_min = market_cap_min if market_cap_min is not None else DataConfig.MARKET_CAP_MIN
        self.market_cap_max = market_cap_max if market_cap_max is not None else DataConfig.MARKET_CAP_MAX
        self.valid_prefixes = valid_prefixes or DataConfig.VALID_STOCK_PREFIXES

    def _get_all_stock_codes(self) -> List[str]:
        """获取 data_all/ 中所有股票代码"""
        codes = []
        for csv_file in sorted(self.data_all_dir.glob("*.csv")):
            code = csv_file.stem
            codes.append(code)
        return codes

    def _filter_by_code(self, stock_codes: List[str]) -> List[str]:
        """第1步：按代码前缀过滤，只保留主板股票"""
        filtered = [code for code in stock_codes
                    if any(code.startswith(prefix) for prefix in self.valid_prefixes)]
        excluded = len(stock_codes) - len(filtered)
        print(f"  [前缀过滤] {len(stock_codes)} → {len(filtered)} (排除 {excluded} 只创业板/科创板/北交所)")
        return filtered

    def _filter_by_active(self, stock_codes: List[str], max_days_behind: int = 30) -> List[str]:
        """第2步：排除退市或长期停牌股票"""
        today = datetime.datetime.now()
        active = []
        inactive_count = 0

        for code in stock_codes:
            file_path = self.data_all_dir / f"{code}.csv"
            if not file_path.exists():
                inactive_count += 1
                continue
            try:
                df = pd.read_csv(file_path, nrows=1)
                if len(df) == 0:
                    inactive_count += 1
                    continue
                latest_date = str(int(df.iloc[0]['date']))
                date_obj = datetime.datetime.strptime(latest_date, "%Y%m%d")
                if (today - date_obj).days <= max_days_behind:
                    active.append(code)
                else:
                    inactive_count += 1
            except Exception:
                inactive_count += 1

        print(f"  [活跃过滤] {len(stock_codes)} → {len(active)} (排除 {inactive_count} 只退市/长期停牌)")
        return active

    def _get_stock_names(self) -> Dict[str, str]:
        """从新浪获取所有A股股票名称（用于判断ST）"""
        print("  [新浪数据] 正在获取A股股票名称...")
        start_time = time.time()
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_spot()
                elapsed = time.time() - start_time
                print(f"  [新浪数据] 获取成功，共 {len(df)} 只股票，耗时 {elapsed:.1f} 秒")
                
                code_to_name = {}
                for _, row in df.iterrows():
                    raw_code = str(row['代码'])
                    if raw_code.startswith('sh') or raw_code.startswith('sz'):
                        code = raw_code[2:].zfill(6)
                    elif raw_code.startswith('bj'):
                        continue
                    else:
                        code = raw_code.zfill(6)
                    name = str(row['名称'])
                    code_to_name[code] = name
                
                return code_to_name
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  [新浪数据] 获取失败 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  [新浪数据] 获取失败: {e}")
                    return {}

    def _calculate_market_cap(self, stock_code: str) -> Optional[float]:
        """
        从本地数据计算流通市值
        
        流通市值 = 成交额 × 100 / 换手率
        
        本地数据字段说明：
        - amount: 成交额（千元）
        - exchange: 换手率（%，流通换手率）
        """
        file_path = self.data_all_dir / f"{stock_code}.csv"
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path, nrows=1)
            if len(df) == 0:
                return None
            
            latest = df.iloc[0]
            amount_k = latest['amount']  # 成交额（千元）
            turnover_pct = latest['exchange']  # 换手率（%）
            
            if pd.isna(amount_k) or pd.isna(turnover_pct) or turnover_pct <= 0:
                return None
            
            amount = amount_k * 1000  # 成交额（元）
            market_cap = amount * 100 / turnover_pct  # 流通市值
            
            return market_cap
        except Exception:
            return None

    def _filter_by_market_cap_and_st(self, stock_codes: List[str]) -> List[str]:
        """第3步：按市值和ST状态过滤"""
        code_to_name = self._get_stock_names()
        
        filtered = []
        st_count = 0
        under_cap_count = 0
        over_cap_count = 0
        no_data_count = 0
        
        for code in stock_codes:
            name = code_to_name.get(code, '')
            
            is_st = 'ST' in name or 'st' in name.lower()
            if is_st:
                st_count += 1
                continue
            
            market_cap = self._calculate_market_cap(code)
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
        
        min_yi = self.market_cap_min / 1e8
        max_yi = self.market_cap_max / 1e8
        print(f"  [市值+ST过滤] {len(stock_codes)} → {len(filtered)} "
              f"(排除 {st_count} 只ST, {under_cap_count} 只市值<{min_yi:.0f}亿, "
              f"{over_cap_count} 只市值>{max_yi:.0f}亿, {no_data_count} 只无数据)")
        return filtered

    def select(self) -> List[str]:
        """执行完整的筛选流程"""
        if not self.data_all_dir.exists():
            print(f"✗ 全量数据目录不存在：{self.data_all_dir}")
            return []

        all_codes = self._get_all_stock_codes()
        if not all_codes:
            print(f"✗ 全量数据目录中无股票数据：{self.data_all_dir}")
            return []

        min_yi = self.market_cap_min / 1e8
        max_yi = self.market_cap_max / 1e8

        print(f"\n{'='*50}")
        print(f"股票数据筛选")
        print(f"{'='*50}")
        print(f"全量数据池：{self.data_all_dir} ({len(all_codes)} 只)")
        print(f"目标目录：{self.data_dir}")
        print(f"市值范围：{min_yi:.0f}亿 ~ {max_yi:.0f}亿（流通市值）")
        print(f"数据源：新浪（股票名称）+ 本地数据（市值计算）")
        print(f"{'='*50}\n")

        codes = self._filter_by_code(all_codes)
        codes = self._filter_by_active(codes)
        codes = self._filter_by_market_cap_and_st(codes)

        print(f"\n{'='*50}")
        print(f"筛选完成：{len(all_codes)} → {len(codes)} 只")
        print(f"{'='*50}")

        return codes

    def copy_to_data(self, selected_codes: List[str]):
        """将筛选后的股票数据复制到 data/ 目录"""
        if self.data_dir.exists():
            for csv_file in self.data_dir.glob("*.csv"):
                csv_file.unlink()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for code in selected_codes:
            src = self.data_all_dir / f"{code}.csv"
            if src.exists():
                shutil.copy2(src, self.data_dir / f"{code}.csv")
                copied += 1

        print(f"\n✓ 已复制 {copied} 只股票到 {self.data_dir}")
        return copied


def main():
    parser = argparse.ArgumentParser(description='股票数据筛选工具 - 从全量数据中筛选符合条件的股票')
    parser.add_argument('--data-all-dir', type=str, default=None,
                        help=f'全量数据目录 (默认：{DataConfig.DATA_ALL_DIR})')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'目标数据目录 (默认：{DataConfig.DATA_DIR})')
    parser.add_argument('--market-cap-min', type=float, default=None,
                        help=f'市值下限/元 (默认：{DataConfig.MARKET_CAP_MIN/1e8:.0f}亿)')
    parser.add_argument('--market-cap-max', type=float, default=None,
                        help=f'市值上限/元 (默认：{DataConfig.MARKET_CAP_MAX/1e8:.0f}亿)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印筛选结果，不复制文件')
    args = parser.parse_args()

    data_all_dir = Path(args.data_all_dir) if args.data_all_dir else Path(DataConfig.DATA_ALL_DIR)
    data_dir = Path(args.data_dir) if args.data_dir else Path(DataConfig.DATA_DIR)

    selector = DataSelector(
        str(data_all_dir), str(data_dir),
        market_cap_min=args.market_cap_min,
        market_cap_max=args.market_cap_max
    )

    selected = selector.select()

    if args.dry_run:
        print(f"\n[Dry Run] 未复制任何文件")
    else:
        selector.copy_to_data(selected)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场指数计算与可视化

从数据库中读取所有 selected 池股票的行情数据，
按日期计算全市场平均涨跌幅，生成 HTML 可视化看板。

用法:
    python src/market_index.py                       # 使用默认数据库路径
    python src/market_index.py --db path/to/db       # 指定数据库
    python src/market_index.py --open                # 生成后自动在浏览器打开
    python src/market_index.py --start 20200101      # 指定起始日期
    python src/market_index.py --end 20251231        # 指定截止日期
"""

import os
import sys
import json
import sqlite3
import argparse
import webbrowser
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'data_maintenance', 'equinet.db')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# ── 数据 JSON 占位符 ──
#   历史 raw 字符串里把 DATA 直接拼进 JS，现在改为把整段 JSON 注入到页面里
#   <script type="application/json" id="market-index-data">…</script> 节点，
#   前端 IIFE 在 init 时 JSON.parse 读取。这样不会再出现"Python raw 串里手写 JS
#   时漏括号"的陷阱（参见修复 market_index tooltip 净值符号不显示的提交）。
#
#   防御：数据 JSON 里若出现字面量 "</script>" 会提前闭合 <script type="application/json">
#   节点，导致后续 HTML 被当成 JS 执行。JSON 允许字符串里出现 "\/"（等价于 "/"），
#   浏览器 JSON.parse 能正确还原，所以这里把 </ 转义为 <\/。
_SAFE_DATA_TOKEN = '__SAFE_DATA_JSON__'

# ── 前端 JS 模板占位符 ──
#   生成时把 src/static/market_index.js 的内容整体内联到 HTML 里，产物仍是单文件，
#   不依赖外部 static/ 资源。这样源码独立（便于语法检查 / 编辑器高亮 / 多人协作），
#   但产物分发时不需要再带一份 static/。
#
#   防御：JS 源里若出现字面量 "</script>" 同样会提前闭合 <script> 节点，
#   所以读入后做一次 </ → <\/ 转义，浏览器 JS 解析器把 <\/ 还原为 </。
_JS_TEMPLATE_TOKEN = '__JS_TEMPLATE__'

# ── JS 模板源文件路径 ──
JS_TEMPLATE_PATH = os.path.join(STATIC_DIR, 'market_index.js')


def compute_market_index(db_path, start_date=None, end_date=None):
    """
    从数据库读取 selected 池股票行情，按日计算全市场平均涨跌幅。

    Returns:
        list[dict]: 每日记录 [{date, yyyymmdd, avg_change, stock_count, ...}, ...]
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT sd.stock_code, sd.date, sd.close
        FROM stock_daily sd
        JOIN stock_pool sp ON sd.stock_code = sp.stock_code
        WHERE sp.pool_type='selected' AND sp.is_active=1
        ORDER BY sd.stock_code, sd.date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("数据库中无数据，请先运行数据更新。")
        return []

    # 按股票分组，计算每只股票的日涨跌幅
    df = df.sort_values(['stock_code', 'date'])
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['change'] = (df['close'] - df['prev_close']) / df['prev_close']

    # 去掉第一行（无前日收盘价）
    df = df.dropna(subset=['change'])

    # 过滤异常值（涨跌幅超过 ±11% 视为异常，与 data.py 涨停过滤阈值一致）
    df = df[(df['change'].abs() <= 0.11)]

    # 日期过滤
    if start_date:
        df = df[df['date'] >= int(start_date)]
    if end_date:
        df = df[df['date'] <= int(end_date)]

    # 按日期分组，计算平均涨跌幅
    grouped = df.groupby('date').agg(
        avg_change=('change', 'mean'),
        stock_count=('stock_code', 'nunique'),
        median_change=('change', 'median'),
        up_count=('change', lambda x: (x > 0).sum()),
        down_count=('change', lambda x: (x < 0).sum()),
        flat_count=('change', lambda x: (x == 0).sum()),
    ).reset_index()

    result = []
    for _, row in grouped.iterrows():
        yyyymmdd = int(row['date'])
        result.append({
            'date': f"{yyyymmdd // 10000}-{(yyyymmdd % 10000) // 100:02d}-{yyyymmdd % 100:02d}",
            'yyyymmdd': yyyymmdd,
            'avg_change': round(float(row['avg_change']) * 100, 4),
            'median_change': round(float(row['median_change']) * 100, 4),
            'stock_count': int(row['stock_count']),
            'up_count': int(row['up_count']),
            'down_count': int(row['down_count']),
            'flat_count': int(row['flat_count']),
        })

    return result


def generate_html(data):
    """生成 HTML 可视化页面，返回 HTML 字符串"""

    data_json = json.dumps(data, ensure_ascii=False)
    # 防御 </script> 提前闭合：</ 转义为 <\/，JSON.parse 还原时等价
    safe_data_json = data_json.replace('</', '<\\/')

    # 读入 JS 模板源文件并做同样的 </ 转义，注入到 <script> 节点里
    with open(JS_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        js_source = f.read()
    safe_js_source = js_source.replace('</', '<\\/')

    html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>市场指数 - 全市场平均涨跌幅</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg:#0d1117; --card-bg:#161b22; --border:#30363d;
            --text:#c9d1d9; --text-secondary:#8b949e;
            --green:#3fb950; --red:#f85149; --gold:#d29922;
            --blue:#58a6ff; --purple:#a371f7; --orange:#f0883e;
        }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{
            background:var(--bg); color:var(--text);
            font-family:'Segoe UI','PingFang SC','Microsoft YaHei','Helvetica Neue',sans-serif;
            min-height:100vh; padding:20px; line-height:1.5;
        }
        .container{max-width:1500px; margin:0 auto;}
        .header{text-align:center; padding:24px 0 20px; border-bottom:1px solid var(--border); margin-bottom:24px;}
        .header h1{
            font-size:1.8rem; font-weight:700; letter-spacing:.5px;
            background:linear-gradient(135deg,#e6edf3,#a5d6ff);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
            margin-bottom:6px;
        }
        .header .subtitle{font-size:.9rem; color:var(--text-secondary); letter-spacing:1px;}
        .header .meta-line{font-size:.78rem; color:var(--text-secondary); margin-top:8px;}

        .stats-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:14px; margin-bottom:24px;}
        .stat-card{
            background:var(--card-bg); border:1px solid var(--border); border-radius:12px;
            padding:16px 18px; text-align:center; transition:all .25s; position:relative; overflow:hidden;
        }
        .stat-card:hover{border-color:#58a6ff55; box-shadow:0 0 20px rgba(88,166,255,.08); transform:translateY(-2px);}
        .stat-card .label{font-size:.78rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:6px;}
        .stat-card .value{font-size:1.7rem; font-weight:700; letter-spacing:.3px;}
        .stat-card .value.positive{color:var(--green);}
        .stat-card .value.negative{color:var(--red);}
        .stat-card .value.neutral{color:#e6edf3;}
        .stat-card .value.highlight{color:var(--gold);}
        .stat-card .sub{font-size:.7rem; color:var(--text-secondary); margin-top:2px;}

        .chart-section{background:var(--card-bg); border:1px solid var(--border); border-radius:14px; padding:20px 18px 14px; margin-bottom:20px;}
        .chart-section .section-title{font-size:.95rem; font-weight:600; letter-spacing:.6px; margin-bottom:10px; color:#e6edf3; display:flex; align-items:center; gap:8px; flex-wrap:wrap;}
        .chart-section .section-title .badge{font-size:.7rem; background:#1f6feb22; color:var(--blue); padding:3px 10px; border-radius:20px; font-weight:500;}
        .chart-wrapper{position:relative; width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch;}
        .chart-wrapper.scrollable canvas{min-width:1600px; height:380px;}
        .chart-wrapper.fixed canvas{width:100%; height:380px;}
        .chart-wrapper::-webkit-scrollbar{height:6px;}
        .chart-wrapper::-webkit-scrollbar-thumb{background:#30363d; border-radius:3px;}
        .legend-hint{font-size:.72rem; color:var(--text-secondary); margin-top:8px; display:flex; flex-wrap:wrap; gap:16px; align-items:center;}
        .legend-hint span{display:inline-flex; align-items:center; gap:5px;}
        .footer-note{text-align:center; font-size:.75rem; color:var(--text-secondary); padding:16px 0; letter-spacing:.5px;}

        .range-controls{display:flex; gap:10px; align-items:center; margin-bottom:20px; flex-wrap:wrap; justify-content:center;}
        .range-controls button{
            background:var(--card-bg); color:var(--text); border:1px solid var(--border);
            padding:6px 16px; border-radius:8px; cursor:pointer; font-size:.82rem;
            transition:all .2s;
        }
        .range-controls button:hover{border-color:var(--blue); color:var(--blue);}
        .range-controls button.active{background:#1f6feb33; border-color:var(--blue); color:var(--blue);}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>市场指数</h1>
            <p class="subtitle">全市场平均涨跌幅 · 等权指数 · 涨跌家数</p>
            <p class="meta-line" id="metaLine"></p>
        </div>

        <div class="range-controls" id="rangeControls"></div>

        <div class="stats-grid" id="statsGrid"></div>

        <div class="chart-section">
            <div class="section-title">逐日平均涨跌幅 <span class="badge">等权指数</span></div>
            <div class="chart-wrapper scrollable"><canvas id="chartChange"></canvas></div>
            <div class="legend-hint">
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#3fb950;"></span> 上涨日</span>
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#f85149;"></span> 下跌日</span>
                <span style="color:#d29922;">━</span> 10日移动平均
                <span style="color:#a371f7;">━</span> 中位数涨跌幅
            </div>
        </div>

        <div class="chart-section">
            <div class="section-title">累计指数曲线 <span class="badge">复利净值</span></div>
            <div class="chart-wrapper fixed"><canvas id="chartCumulative"></canvas></div>
            <div class="legend-hint">
                <span style="color:#3fb950;">━</span> 复利净值（初始=1）
                <span style="color:#58a6ff;">━</span> 算术累计涨跌幅%
                <span style="color:#8b949e;">--</span> 净值=1 基准线
            </div>
        </div>

        <div class="chart-section">
            <div class="section-title">涨跌家数 <span class="badge">市场宽度</span></div>
            <div class="chart-wrapper scrollable"><canvas id="chartUpDown"></canvas></div>
            <div class="legend-hint">
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#3fb950;"></span> 上涨</span>
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#f85149;"></span> 下跌</span>
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#8b949e;"></span> 平盘</span>
                <span style="color:#d29922;">━</span> 涨跌比（上涨/下跌）
            </div>
        </div>

        <div class="chart-section">
            <div class="section-title">回撤分析 <span class="badge">基于复利净值</span></div>
            <div class="chart-wrapper fixed"><canvas id="chartDrawdown"></canvas></div>
            <div class="legend-hint">
                <span style="color:#f85149;">━</span> 当前回撤
                <span style="color:#f0883e;">--</span> 最大回撤线
            </div>
        </div>

        <div class="footer-note">
            等权指数 = 每日所有股票涨跌幅的算术平均 | 共 <strong id="totalDaysFoot">--</strong> 个交易日
        </div>
    </div>

    <!-- 节点顺序约束：JS 模板必须在数据节点之前。
         模板 IIFE 在脚本解析阶段同步执行，立即 document.getElementById('market-index-data')
         读取数据节点；颠倒顺序会让 IIFE 拿到 null、fallback 为空数组，图表静默失效。
         调换顺序前请同步修改 src/static/market_index.js 的读取时机（包一层 DOMContentLoaded）。 -->
    <script>''' + _JS_TEMPLATE_TOKEN + r'''</script>
    <script type="application/json" id="market-index-data">''' + _SAFE_DATA_TOKEN + r'''</script>
</body>
</html>''';

    return (html
            .replace(_JS_TEMPLATE_TOKEN, safe_js_source)
            .replace(_SAFE_DATA_TOKEN, safe_data_json))


def main():
    parser = argparse.ArgumentParser(description='市场指数 - 全市场平均涨跌幅可视化')
    parser.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--start', type=int, default=None, help='起始日期 YYYYMMDD')
    parser.add_argument('--end', type=int, default=None, help='截止日期 YYYYMMDD')
    parser.add_argument('--open', action='store_true', help='生成后自动在浏览器打开')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"数据库不存在: {args.db}")
        sys.exit(1)

    print("正在计算市场指数...")
    data = compute_market_index(args.db, args.start, args.end)

    if not data:
        print("无有效数据，退出。")
        sys.exit(1)

    print(f"共 {len(data)} 个交易日，区间: {data[0]['date']} ~ {data[-1]['date']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'market_index.html')

    html = generate_html(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"已生成: {output_path}")

    # 保存每日数据到 JSON，供训练时做极端行情过滤
    json_path = os.path.join(OUTPUT_DIR, 'market_index.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"数据已保存: {json_path}")

    if args.open:
        webbrowser.open('file://' + os.path.abspath(output_path))


if __name__ == '__main__':
    main()

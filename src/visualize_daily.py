#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取各训练 run 目录下的每日统计 JSON，生成 HTML 可视化看板。

目录即模型：每个模型的 daily_stats_<时间戳>.json 由 run.py 写入该模型
所在的 run 目录（out/<日期戳>/），本脚本按此布局扫描。

用法:
    python src/visualize_daily.py                       # 自动取 out/ 各 run 目录下最新的 daily_stats_*.json
    python src/visualize_daily.py path/to/xxx.json      # 指定 JSON
    python src/visualize_daily.py xxx.json --open       # 生成后自动在浏览器打开
"""
import os
import sys
import json
import glob
import argparse
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 每日统计 JSON 的家是各 run 目录（out/<日期戳>/），这里只是"默认扫描根"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# ── 占位符（替换顺序无关：两个 token 字符串互不相同） ──
#   数据 JSON 与 META JSON 注入到 <script type="application/json"> 节点，
#   前端 IIFE 在 init 时 JSON.parse 读取，避免把 JSON 直接拼进 JS 上下文
#   （参见重构 market_index 的同款注释）。
_SAFE_DATA_TOKEN = '__SAFE_DAILY_DATA_JSON__'
_SAFE_META_TOKEN = '__SAFE_DAILY_META_JSON__'

# ── 前端 JS 模板占位符 ──
#   生成时把 src/static/visualize_daily.js 的内容整体内联到 HTML，
#   产物仍是单文件，不依赖外部 static/ 资源。
#
#   防御：JSON 与 JS 源里若出现字面量 "</script>" 会提前闭合对应的
#   <script> 节点，所以读入后统一做 </ → <\/ 转义；JSON.parse 与
#   JS 词法解析都会把 <\/ 还原成 /。
_JS_TEMPLATE_TOKEN = '__DAILY_JS_TEMPLATE__'

# ── JS 模板源文件路径 ──
JS_TEMPLATE_PATH = os.path.join(STATIC_DIR, 'visualize_daily.js')


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型赚钱效应分析 - 每日选股收益率</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg:#0d1117; --card-bg:#161b22; --border:#30363d;
            --text:#c9d1d9; --text-secondary:#8b949e;
            --green:#3fb950; --red:#f85149; --gold:#d29922;
            --blue:#58a6ff; --purple:#a371f7; --orange:#f0883e; --teal:#39d353;
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
        .chart-section .section-title .fix{font-size:.68rem; background:#f0883e22; color:var(--orange); padding:3px 10px; border-radius:20px;}
        .chart-wrapper{position:relative; width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch;}
        .chart-wrapper.scrollable canvas{min-width:1600px; height:380px;}
        .chart-wrapper.fixed canvas{width:100%; height:380px;}
        .chart-wrapper::-webkit-scrollbar{height:6px;}
        .chart-wrapper::-webkit-scrollbar-thumb{background:#30363d; border-radius:3px;}
        .legend-hint{font-size:.72rem; color:var(--text-secondary); margin-top:8px; display:flex; flex-wrap:wrap; gap:16px; align-items:center;}
        .legend-hint span{display:inline-flex; align-items:center; gap:5px;}
        .footer-note{text-align:center; font-size:.75rem; color:var(--text-secondary); padding:16px 0; letter-spacing:.5px;}
        @media (max-width:768px){
            .stats-grid{grid-template-columns:repeat(3,1fr); gap:8px;}
            .stat-card{padding:10px 8px;} .stat-card .value{font-size:1.25rem;}
            .chart-wrapper.scrollable canvas{min-width:1200px; height:280px;}
            .chart-wrapper.fixed canvas{height:280px;}
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 模型赚钱效应分析</h1>
            <p class="subtitle">逐日选股收益率 · 累计收益曲线 · 回撤（已修正）</p>
            <p class="meta-line" id="metaLine"></p>
        </div>

        <div class="stats-grid" id="statsGrid"></div>

        <div class="chart-section">
            <div class="section-title">📈 逐日选股收益率 <span class="badge">每日盈亏%</span></div>
            <div class="chart-wrapper scrollable"><canvas id="chartReturns"></canvas></div>
            <div class="legend-hint">
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#3fb950;"></span> 盈利日</span>
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#f85149;"></span> 亏损日</span>
                <span style="color:#d29922;">━</span> 10日移动平均
                <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f0883e;"></span> 选股数不足（count &lt; 上限）</span>
            </div>
        </div>

        <div class="chart-section">
            <div class="section-title">💰 累计收益率曲线 <span class="badge">赚钱效应</span></div>
            <div class="chart-wrapper fixed"><canvas id="chartCumulative"></canvas></div>
            <div class="legend-hint">
                <span style="color:#3fb950;">━</span> 复利累计（等权组合）
                <span style="color:#58a6ff;">━</span> 算术累加（后端口径）
                <span style="color:#8b949e;">--</span> 零轴
            </div>
        </div>

        <div class="chart-section">
            <div class="section-title">📉 回撤分析 <span class="fix">基于复利资金曲线（已修正原文件 bug）</span></div>
            <div class="chart-wrapper fixed"><canvas id="chartDrawdown"></canvas></div>
            <div class="legend-hint">
                <span style="color:#f85149;">━</span> 当前回撤
                <span style="color:#f0883e;">--</span> 最大回撤线
            </div>
        </div>

        <div class="footer-note">
            💡 复利曲线 = 每日等权持有选中股票的净值复利（不含持仓上限/T+1 等资金管理约束，与逐日交易明细口径不同） | 共 <strong id="totalDaysFoot">--</strong> 个有效交易日
        </div>
    </div>

    <!-- 节点顺序约束：两个 <script type="application/json"> 数据节点必须在 JS 模板之前。
         模板 IIFE 在脚本解析阶段同步执行，立即 document.getElementById('daily-data' / 'daily-meta')
         读取数据节点；颠倒顺序会让 IIFE 拿到 null、fallback 为空数组 / {}，图表与 meta 行静默失效。
         调换顺序前请同步修改 src/static/visualize_daily.js 的读取时机（包一层 DOMContentLoaded）。 -->
    <script type="application/json" id="daily-data">''' + _SAFE_DATA_TOKEN + r'''</script>
    <script type="application/json" id="daily-meta">''' + _SAFE_META_TOKEN + r'''</script>
    <script>''' + _JS_TEMPLATE_TOKEN + r'''</script>
</body>
</html>
'''


def find_latest_json(directory=DEFAULT_OUTPUT_DIR):
    """在 directory 的各 run 目录下找最新的 daily_stats_*.json。

    目录即模型：daily_stats 只会出现在 out/<日期戳>/ 里，扫一层子目录即可。
    """
    files = glob.glob(os.path.join(directory, '*', 'daily_stats_*.json'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_payload(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _escape_script_payload(text):
    r"""把 ``</`` 转义为 ``<\/``，防御 ``</script>`` 提前闭合当前 <script> 节点。

    浏览器 JS 引擎与 JSON.parse 都把 ``<\/`` 视为 ``/`` 的等价表达，所以对合法
    JSON 与 JS 都是无损的。
    """
    return text.replace('</', '<\\/')


def _load_js_template():
    with open(JS_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def render_html(payload):
    data = payload.get('daily', [])
    meta = payload.get('meta', {})

    safe_data = _escape_script_payload(json.dumps(data, ensure_ascii=False))
    safe_meta = _escape_script_payload(json.dumps(meta, ensure_ascii=False))
    safe_js = _escape_script_payload(_load_js_template())

    return (HTML_TEMPLATE
            .replace(_JS_TEMPLATE_TOKEN, safe_js)
            .replace(_SAFE_DATA_TOKEN, safe_data)
            .replace(_SAFE_META_TOKEN, safe_meta))


def main():
    parser = argparse.ArgumentParser(description='读取各 run 目录下每日统计 JSON 生成 HTML 可视化')
    parser.add_argument('json_path', nargs='?', default=None,
                        help='daily_stats_*.json 路径；省略则自动取 out/ 各 run 目录下最新一个')
    parser.add_argument('--open', action='store_true', help='生成后在默认浏览器打开')
    parser.add_argument('--out', default=None, help='输出 HTML 路径（默认与 JSON 同目录）')
    args = parser.parse_args()

    json_path = args.json_path or find_latest_json()
    if not json_path or not os.path.exists(json_path):
        print('✗ 未找到每日统计 JSON。请先运行: python src/run.py [--begin YYYYMMDD]')
        print('  再用: python src/visualize_daily.py <json路径>')
        sys.exit(1)

    payload = load_payload(json_path)
    n_days = len(payload.get('daily', []))
    if n_days == 0:
        print(f'✗ {json_path} 中没有每日数据（daily 为空），无法可视化。')
        sys.exit(1)

    html = render_html(payload)

    if args.out:
        out_path = args.out
    else:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        # 跟随 JSON 所在目录：HTML 与其数据同住一个家
        out_dir = os.path.dirname(os.path.abspath(json_path)) or DEFAULT_OUTPUT_DIR
        out_path = os.path.join(out_dir, f'{stem}_dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'✓ 读取数据: {json_path}（{n_days} 个交易日）')
    print(f'✓ 可视化已生成: {out_path}')
    if args.open:
        webbrowser.open('file:///' + os.path.abspath(out_path).replace('\\', '/'))
        print('  已在默认浏览器打开。')


if __name__ == '__main__':
    main()

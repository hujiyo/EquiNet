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

# ── 唯一占位符，避免 f-string 与 CSS/JS 大括号冲突 ──
_DATA_TOKEN = '__DATA_JSON__'


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

    import json
    data_json = json.dumps(data, ensure_ascii=False)

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

    <script>
    (function(){
        const DATA = ''' + _DATA_TOKEN + r''';
        const daily = (DATA || []).slice().sort((a,b)=>a.yyyymmdd-b.yyyymmdd);
        const N = daily.length;
        const labels = daily.map(d=>d.date);
        const avgChanges = daily.map(d=>d.avg_change);
        const medianChanges = daily.map(d=>d.median_change);
        const stockCounts = daily.map(d=>d.stock_count);
        const upCounts = daily.map(d=>d.up_count);
        const downCounts = daily.map(d=>d.down_count);
        const flatCounts = daily.map(d=>d.flat_count);

        // ===== 复利净值 & 算术累计 =====
        const equity = [];
        const cumArith = [];
        let prod = 1, sumA = 0;
        for (let i=0;i<N;i++){
            prod *= (1 + avgChanges[i]/100);
            sumA += avgChanges[i];
            equity.push(prod);
            cumArith.push(+sumA.toFixed(4));
        }

        // ===== 10日移动平均 =====
        const ma10 = [];
        for (let i=0;i<N;i++){
            const s = Math.max(0, i-9);
            const slice = avgChanges.slice(s, i+1);
            ma10.push(+(slice.reduce((a,b)=>a+b,0)/slice.length).toFixed(4));
        }

        // ===== 回撤 =====
        const drawdowns = [];
        let peakEq = equity.length ? equity[0] : 1;
        for (let i=0;i<N;i++){
            if (equity[i] > peakEq) peakEq = equity[i];
            const dd = peakEq > 0 ? (equity[i]-peakEq)/peakEq*100 : 0;
            drawdowns.push(+dd.toFixed(4));
        }
        const maxDrawdown = N ? Math.min.apply(null, drawdowns) : 0;

        // ===== 涨跌比 =====
        const upDownRatio = daily.map(d => d.down_count > 0 ? +(d.up_count / d.down_count).toFixed(2) : (d.up_count > 0 ? 99 : 0));

        // ===== 统计量 =====
        const winCount = avgChanges.filter(r=>r>0).length;
        const lossCount = avgChanges.filter(r=>r<0).length;
        const winRate = N ? winCount/N*100 : 0;
        const totalArith = N ? cumArith[N-1] : 0;
        const totalCompound = N ? +((equity[N-1]-1)*100).toFixed(4) : 0;
        const avgReturn = N ? avgChanges.reduce((a,b)=>a+b,0)/N : 0;
        const maxSingleUp = N ? Math.max.apply(null, avgChanges) : 0;
        const maxSingleDown = N ? Math.min.apply(null, avgChanges) : 0;
        const variance = N ? avgChanges.reduce((s,r)=>s+(r-avgReturn)*(r-avgReturn),0)/N : 0;
        const stdDev = Math.sqrt(variance);
        const sharpeApprox = stdDev>0 ? (avgReturn/stdDev)*Math.sqrt(N) : 0;
        const avgStockCount = N ? stockCounts.reduce((a,b)=>a+b,0)/N : 0;

        // ===== Meta 行 =====
        const startDate = N ? daily[0].date : '?';
        const endDate = N ? daily[N-1].date : '?';
        document.getElementById('metaLine').textContent =
            '区间: ' + startDate + ' ~ ' + endDate + '  |  平均覆盖: ' + avgStockCount.toFixed(0) + ' 只股票/日';
        document.getElementById('totalDaysFoot').textContent = N;

        // ===== 时间范围快捷按钮 =====
        const rangeDiv = document.getElementById('rangeControls');
        const ranges = [
            {label:'全部', months:0},
            {label:'近1年', months:12},
            {label:'近6月', months:6},
            {label:'近3月', months:3},
        ];
        // 存储原始数据供范围切换
        window._fullData = {daily, N, labels, avgChanges, medianChanges, stockCounts, upCounts, downCounts, flatCounts, equity, cumArith, ma10, drawdowns, upDownRatio};

        ranges.forEach((r, idx) => {
            const btn = document.createElement('button');
            btn.textContent = r.label;
            if (idx === 0) btn.classList.add('active');
            btn.addEventListener('click', function(){
                rangeDiv.querySelectorAll('button').forEach(b=>b.classList.remove('active'));
                btn.classList.add('active');
                applyRange(r.months);
            });
            rangeDiv.appendChild(btn);
        });

        function applyRange(months) {
            // months=0 means all
            const F = window._fullData;
            let startIdx = 0;
            if (months > 0 && F.N > 0) {
                const lastDate = F.daily[F.N-1].yyyymmdd;
                const y = Math.floor(lastDate / 10000);
                const m = Math.floor((lastDate % 10000) / 100);
                const d = lastDate % 100;
                let targetM = m - months;
                let targetY = y;
                while (targetM <= 0) { targetM += 12; targetY--; }
                const targetDate = targetY * 10000 + targetM * 100 + d;
                startIdx = F.daily.findIndex(dd => dd.yyyymmdd >= targetDate);
                if (startIdx < 0) startIdx = 0;
            }
            updateCharts(startIdx);
        }

        // ===== 统计卡片 =====
        function renderStats(startIdx) {
            const F = window._fullData;
            const n = F.N - startIdx;
            if (n <= 0) return;
            const changes = F.avgChanges.slice(startIdx);
            const eq = F.equity.slice(startIdx);
            const counts = F.stockCounts.slice(startIdx);
            const dd = F.drawdowns.slice(startIdx);
            const cumA = F.cumArith.slice(startIdx);

            const wc = changes.filter(r=>r>0).length;
            const lc = changes.filter(r=>r<0).length;
            const wr = wc/n*100;
            const ta = cumA[cumA.length-1];
            const tc = +((eq[eq.length-1]-1)*100).toFixed(4);
            const ar = changes.reduce((a,b)=>a+b,0)/n;
            const maxUp = Math.max.apply(null, changes);
            const maxDown = Math.min.apply(null, changes);
            const v = changes.reduce((s,r)=>s+(r-ar)*(r-ar),0)/n;
            const sd = Math.sqrt(v);
            const sh = sd>0 ? (ar/sd)*Math.sqrt(n) : 0;
            const mdd = Math.min.apply(null, dd);
            const asc = counts.reduce((a,b)=>a+b,0)/n;

            const fmtPct = v => (v>=0?'+':'') + (+v).toFixed(2) + '%';
            const fmtNum = (v,d=2) => (+v).toFixed(d);
            const cls = (cond, negCond) => cond?'positive':(negCond?'negative':'neutral');
            const cards = [
                {label:'交易日', value:String(n), cls:'neutral', sub:'有效交易天数'},
                {label:'复利累计', value:fmtPct(tc), cls:cls(tc>=0,true), sub:'等权组合复利'},
                {label:'算术累计', value:fmtPct(ta), cls:cls(ta>=0,true), sub:'Σ 每日平均'},
                {label:'上涨日占比', value:fmtNum(wr,1)+'%', cls:cls(wr>=50,true), sub:wc+'涨/'+lc+'跌'},
                {label:'日均涨跌幅', value:fmtPct(ar), cls:cls(ar>=0,true), sub:'期望值'},
                {label:'最大单日涨幅', value:'+'+fmtNum(maxUp,2)+'%', cls:'highlight', sub:'最佳交易日'},
                {label:'最大单日跌幅', value:fmtNum(maxDown,2)+'%', cls:'negative', sub:'最差交易日'},
                {label:'最大回撤', value:fmtNum(mdd,2)+'%', cls:'negative', sub:'基于复利净值'},
                {label:'夏普近似', value:fmtNum(sh,2), cls:sh>=1?'positive':'neutral', sub:'风险调整收益'},
                {label:'日均股票数', value:fmtNum(asc,0), cls:'neutral', sub:'覆盖股票只数'},
            ];
            document.getElementById('statsGrid').innerHTML = cards.map(c=>(
                '<div class="stat-card"><div class="label">'+c.label+'</div>'+
                '<div class="value '+c.cls+'">'+c.value+'</div><div class="sub">'+c.sub+'</div></div>'
            )).join('');
        }

        // ===== Chart 实例存储 =====
        let chartChange, chartCumulative, chartUpDown, chartDrawdown;

        function updateCharts(startIdx) {
            const F = window._fullData;
            const sl = F.labels.slice(startIdx);
            const sc = F.avgChanges.slice(startIdx);
            const sm = F.medianChanges.slice(startIdx);
            const sma = F.ma10.slice(startIdx);
            const seq = F.equity.slice(startIdx);
            const sca = F.cumArith.slice(startIdx);
            const su = F.upCounts.slice(startIdx);
            const sd2 = F.downCounts.slice(startIdx);
            const sf = F.flatCounts.slice(startIdx);
            const sdd = F.drawdowns.slice(startIdx);
            const sr = F.upDownRatio.slice(startIdx);
            const ssc = F.stockCounts.slice(startIdx);

            const mdd = sdd.length ? Math.min.apply(null, sdd) : 0;

            renderStats(startIdx);

            // ── 图1: 逐日涨跌幅 ──
            const barColors = sc.map(r=>r>=0?'rgba(63,185,80,0.75)':'rgba(248,81,73,0.75)');
            const barBorders = sc.map(r=>r>=0?'rgba(63,185,80,0.95)':'rgba(248,81,73,0.95)');

            if (chartChange) chartChange.destroy();
            chartChange = new Chart(document.getElementById('chartChange').getContext('2d'), {
                type:'bar',
                data:{labels:sl, datasets:[
                    {label:'平均涨跌幅 %', data:sc, backgroundColor:barColors, borderColor:barBorders,
                     borderWidth:0.3, borderRadius:1, barPercentage:0.85, categoryPercentage:0.9, order:1},
                    {label:'10日移动平均', data:sma, type:'line', borderColor:'#d29922',
                     backgroundColor:'rgba(210,153,34,0.08)', borderWidth:1.8, pointRadius:0, tension:0.35, fill:false, order:0},
                    {label:'中位数涨跌幅', data:sm, type:'line', borderColor:'#a371f7',
                     backgroundColor:'transparent', borderWidth:1.2, borderDash:[4,3], pointRadius:0, tension:0.35, fill:false, order:-1},
                ]},
                options:{
                    responsive:true, maintainAspectRatio:false,
                    interaction:{mode:'index', intersect:false},
                    plugins:{
                        legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                        tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                            borderColor:'#58a6ff', borderWidth:1, padding:10,
                            callbacks:{
                                label:c=>{
                                    const i=c.dataIndex;
                                    if(c.datasetIndex===0) return '均值: '+(sc[i]>=0?'+':'')+sc[i].toFixed(2)+'% | 股票: '+ssc[i];
                                    if(c.datasetIndex===1) return 'MA10: '+(sma[i]>=0?'+':'')+sma[i].toFixed(2)+'%';
                                    if(c.datasetIndex===2) return '中位数: '+(sm[i]>=0?'+':'')+sm[i].toFixed(2)+'%';
                                    return '';
                                }
                            }
                        }
                    },
                    scales:{
                        x:{ticks:{color:'#8b949e', maxTicksLimit:40, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                        y:{title:{display:true, text:'涨跌幅 (%)', color:'#8b949e'},
                           ticks:{color:'#8b949e', callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'}, grid:{color:'#30363d55'}}
                    }
                }
            });

            // ── 图2: 累计指数 ──
            const cumCompound = seq.map(e=>+((e-1)*100).toFixed(4));
            if (chartCumulative) chartCumulative.destroy();
            chartCumulative = new Chart(document.getElementById('chartCumulative').getContext('2d'), {
                type:'line',
                data:{labels:sl, datasets:[
                    {label:'复利净值', data:seq, borderColor:'#3fb950',
                     backgroundColor:'rgba(63,185,80,0.07)', borderWidth:2.2, pointRadius:0, tension:0.3, fill:true,
                     yAxisID:'y', order:0},
                    {label:'算术累计 %', data:sca, borderColor:'#58a6ff',
                     backgroundColor:'transparent', borderWidth:1.6, borderDash:[6,4], pointRadius:0, tension:0.3, fill:false,
                     yAxisID:'y1', order:1},
                ]},
                options:{
                    responsive:true, maintainAspectRatio:false,
                    interaction:{mode:'index', intersect:false},
                    plugins:{
                        legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                        tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                            borderColor:'#3fb950', borderWidth:1, padding:10,
                            callbacks:{label:c=>c.datasetIndex===0?
                                '净值: '+seq[c.dataIndex].toFixed(4)+' ('+cumCompound[c.dataIndex]>=0?'+':''+cumCompound[c.dataIndex].toFixed(2)+'%)':
                                '算术累计: '+(sca[c.dataIndex]>=0?'+':'')+sca[c.dataIndex].toFixed(2)+'%'}}
                    },
                    scales:{
                        x:{ticks:{color:'#8b949e', maxTicksLimit:30, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                        y:{type:'linear', position:'left', title:{display:true, text:'复利净值', color:'#3fb950'},
                           ticks:{color:'#3fb950', callback:v=>v.toFixed(2)}, grid:{color:'#30363d55'}},
                        y1:{type:'linear', position:'right', title:{display:true, text:'算术累计 %', color:'#58a6ff'},
                            ticks:{color:'#58a6ff', callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'}, grid:{drawOnChartArea:false}},
                    }
                },
                plugins:[{id:'baseLine', beforeDraw(ch){
                    const {ctx, scales}=ch; const y0=scales.y.getPixelForValue(1);
                    ctx.save(); ctx.strokeStyle='rgba(230,237,243,0.45)'; ctx.lineWidth=1.3; ctx.setLineDash([8,4]);
                    ctx.beginPath(); ctx.moveTo(scales.x.left,y0); ctx.lineTo(scales.x.right,y0); ctx.stroke(); ctx.setLineDash([]); ctx.restore();
                }}]
            });

            // ── 图3: 涨跌家数 ──
            if (chartUpDown) chartUpDown.destroy();
            chartUpDown = new Chart(document.getElementById('chartUpDown').getContext('2d'), {
                type:'bar',
                data:{labels:sl, datasets:[
                    {label:'上涨', data:su, backgroundColor:'rgba(63,185,80,0.7)', borderColor:'rgba(63,185,80,0.9)',
                     borderWidth:0.3, borderRadius:1, stack:'stack0', order:2},
                    {label:'下跌', data:sd2.map(v=>-v), backgroundColor:'rgba(248,81,73,0.7)', borderColor:'rgba(248,81,73,0.9)',
                     borderWidth:0.3, borderRadius:1, stack:'stack0', order:2},
                    {label:'平盘', data:sf, backgroundColor:'rgba(139,148,158,0.5)', borderColor:'rgba(139,148,158,0.7)',
                     borderWidth:0.3, borderRadius:1, stack:'stack1', order:3},
                    {label:'涨跌比', data:sr, type:'line', borderColor:'#d29922',
                     backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false,
                     yAxisID:'y1', order:0},
                ]},
                options:{
                    responsive:true, maintainAspectRatio:false,
                    interaction:{mode:'index', intersect:false},
                    plugins:{
                        legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                        tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                            borderColor:'#d29922', borderWidth:1, padding:10,
                            callbacks:{label:c=>{
                                const i=c.dataIndex;
                                if(c.datasetIndex===0) return '上涨: '+su[i];
                                if(c.datasetIndex===1) return '下跌: '+sd2[i];
                                if(c.datasetIndex===2) return '平盘: '+sf[i];
                                if(c.datasetIndex===3) return '涨跌比: '+sr[i].toFixed(2);
                                return '';
                            }}
                        }
                    },
                    scales:{
                        x:{stacked:true, ticks:{color:'#8b949e', maxTicksLimit:40, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                        y:{stacked:true, title:{display:true, text:'家数', color:'#8b949e'},
                           ticks:{color:'#8b949e'}, grid:{color:'#30363d55'}},
                        y1:{type:'linear', position:'right', title:{display:true, text:'涨跌比', color:'#d29922'},
                            ticks:{color:'#d29922'}, grid:{drawOnChartArea:false}, min:0},
                    }
                }
            });

            // ── 图4: 回撤 ──
            if (chartDrawdown) chartDrawdown.destroy();
            chartDrawdown = new Chart(document.getElementById('chartDrawdown').getContext('2d'), {
                type:'line',
                data:{labels:sl, datasets:[
                    {label:'回撤 %', data:sdd, borderColor:'#f85149',
                     backgroundColor:'rgba(248,81,73,0.10)', borderWidth:1.8, pointRadius:0, tension:0.25, fill:true},
                    {label:'最大回撤线', data:Array(sdd.length).fill(mdd), borderColor:'#f0883e',
                     backgroundColor:'transparent', borderWidth:1.2, borderDash:[8,5], pointRadius:0, tension:0, fill:false},
                ]},
                options:{
                    responsive:true, maintainAspectRatio:false,
                    interaction:{mode:'index', intersect:false},
                    plugins:{
                        legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                        tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                            borderColor:'#f85149', borderWidth:1, padding:10,
                            callbacks:{label:c=>c.datasetIndex===0?
                                '回撤: '+sdd[c.dataIndex].toFixed(2)+'%':'最大回撤: '+mdd.toFixed(2)+'%'}}
                    },
                    scales:{
                        x:{ticks:{color:'#8b949e', maxTicksLimit:30, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                        y:{title:{display:true, text:'回撤 (%)', color:'#8b949e'},
                           ticks:{color:'#8b949e', callback:v=>v.toFixed(1)+'%'}, grid:{color:'#30363d55'}, max:0}
                    }
                }
            });
        }

        // 初始渲染
        updateCharts(0);

    })();
    </script>
</body>
</html>''';

    return html.replace(_DATA_TOKEN, data_json)


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

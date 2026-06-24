#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 out_run/ 下的每日统计 JSON，生成 HTML 可视化看板。

用法:
    python src/visualize_daily.py                       # 自动取 out_run/ 下最新的 daily_stats_*.json
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
OUT_RUN_DIR = os.path.join(PROJECT_ROOT, 'out_run')

# 用唯一占位符 + str.replace 注入数据，避免 f-string/format 与 CSS/JS 的大括号冲突
_DATA_TOKEN = '__DATA_JSON__'
_META_TOKEN = '__META_JSON__'

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

    <script>
    (function(){
        const DATA = __DATA_JSON__;
        const META = __META_JSON__;

        // ===== 基础序列 =====
        const daily = (DATA || []).slice().sort((a,b)=>a.yyyymmdd-b.yyyymmdd);
        const N = daily.length;
        const labels = daily.map(d=>d.date);
        const returns = daily.map(d=>d.return_pct);     // 每日收益率 %
        const counts  = daily.map(d=>d.count);
        // 选股上限：top_n 模式用 TOP_N_PER_DAY，全局阈值模式用 MAX_SELECT_PER_DAY；
        // 任一为 0/未配置（不限数量）时无"选股不足"概念，underSelectIdx 自然为空
        const maxSelect = (META.mode === 'top_n_per_day' && META.top_n_per_day > 0)
            ? META.top_n_per_day
            : (META.max_select_per_day > 0 ? META.max_select_per_day : 0);

        // ===== 复利资金曲线（equity，初始=1）与算术累加 =====
        const equity = [];      // 复利资金净值
        const cumArith = [];    // 算术累加 %
        const cumCompound = []; // 复利累计收益 %
        let prod = 1, sumA = 0;
        for (let i=0;i<N;i++){
            prod *= (1 + returns[i]/100);
            sumA += returns[i];
            equity.push(prod);
            cumArith.push(+sumA.toFixed(4));
            cumCompound.push(+((prod-1)*100).toFixed(4));
        }

        // ===== 10日移动平均 =====
        const ma10 = [];
        for (let i=0;i<N;i++){
            const s = Math.max(0, i-9);
            const slice = returns.slice(s, i+1);
            ma10.push(+(slice.reduce((a,b)=>a+b,0)/slice.length).toFixed(4));
        }

        // ===== 回撤：在复利资金曲线 equity 上计算标准相对回撤 =====
        // 修正点：原文件在"累计百分比"序列上做 (cur-peak)/peak，peak≈0 时失真。
        // 这里 equity>=0、peakEquity>=1，除法恒良定义。
        const drawdowns = [];
        let peakEq = equity.length ? equity[0] : 1;
        for (let i=0;i<N;i++){
            if (equity[i] > peakEq) peakEq = equity[i];
            const dd = peakEq > 0 ? (equity[i]-peakEq)/peakEq*100 : 0;
            drawdowns.push(+dd.toFixed(4));
        }
        const maxDrawdown = N ? Math.min.apply(null, drawdowns) : 0;
        const maxDrawdownIdx = drawdowns.indexOf(maxDrawdown);

        // ===== 统计量 =====
        const winCount = returns.filter(r=>r>0).length;
        const lossCount = returns.filter(r=>r<0).length;
        const flatCount = returns.filter(r=>r===0).length;
        const winRate = N ? winCount/N*100 : 0;
        const totalArith = N ? cumArith[N-1] : 0;
        const totalCompound = N ? cumCompound[N-1] : 0;
        const avgReturn = N ? returns.reduce((a,b)=>a+b,0)/N : 0;
        const maxSingleWin = N ? Math.max.apply(null, returns) : 0;
        const maxSingleLoss = N ? Math.min.apply(null, returns) : 0;
        const variance = N ? returns.reduce((s,r)=>s+(r-avgReturn)*(r-avgReturn),0)/N : 0;
        const stdDev = Math.sqrt(variance);
        const sharpeApprox = stdDev>0 ? (avgReturn/stdDev)*Math.sqrt(N) : 0;
        const avgWin = winCount>0 ? returns.filter(r=>r>0).reduce((a,b)=>a+b,0)/winCount : 0;
        const avgLoss = lossCount>0 ? Math.abs(returns.filter(r=>r<0).reduce((a,b)=>a+b,0)/lossCount) : 0;
        const profitLossRatio = avgLoss>0 ? avgWin/avgLoss : (avgWin>0?Infinity:0);
        const avgCount = N ? counts.reduce((a,b)=>a+b,0)/N : 0;
        const underSelectIdx = counts.map((c,i)=>c<maxSelect?i:-1).filter(i=>i>=0);

        // ===== 顶部 meta 行 =====
        document.getElementById('metaLine').textContent =
            '模型: ' + (META.model||'?') + '  |  区间: ' + (META.start_date||'?') + ' ~ ' + (META.end_date||'?') +
            (META.begin_date ? '  |  --begin: '+META.begin_date : '') +
            '  |  阈值: ' + (META.threshold!=null? (+META.threshold).toFixed(6):'?');
        document.getElementById('totalDaysFoot').textContent = N;

        // ===== 统计卡片 =====
        const fmtPct = v => (v>=0?'+':'') + (+v).toFixed(2) + '%';
        const fmtNum = (v,d=2) => (+v).toFixed(d);
        const cls = (cond, negCond) => cond?'positive':(negCond?'negative':'neutral');
        const cards = [
            {label:'有效交易日', value:String(N), cls:'neutral', sub:'count>0 的日子'},
            {label:'复利累计收益', value:fmtPct(totalCompound), cls:cls(totalCompound>=0,true), sub:'等权组合复利'},
            {label:'算术累计收益', value:fmtPct(totalArith), cls:cls(totalArith>=0,true), sub:'Σ 每日(后端口径)'},
            {label:'胜率', value:fmtNum(winRate,1)+'%', cls:cls(winRate>=50,true), sub:winCount+'赢/'+lossCount+'亏/'+flatCount+'平'},
            {label:'平均每日收益', value:fmtPct(avgReturn), cls:cls(avgReturn>=0,true), sub:'期望值'},
            {label:'最大单日盈利', value:'+'+fmtNum(maxSingleWin,1)+'%', cls:'highlight', sub:'最佳交易日'},
            {label:'最大单日亏损', value:fmtNum(maxSingleLoss,1)+'%', cls:'negative', sub:'最差交易日'},
            {label:'最大回撤', value:fmtNum(maxDrawdown,2)+'%', cls:'negative', sub:maxDrawdownIdx>=0?('第'+(maxDrawdownIdx+1)+'日附近'):'-'},
            {label:'盈亏比', value:isFinite(profitLossRatio)?fmtNum(profitLossRatio,2):'∞', cls:profitLossRatio>=2?'positive':'neutral', sub:'avgWin/avgLoss'},
            {label:'夏普近似', value:fmtNum(sharpeApprox,2), cls:sharpeApprox>=1?'positive':'neutral', sub:'风险调整收益'},
            {label:'平均选股数', value:fmtNum(avgCount,1), cls:'neutral', sub:'每日入选只数'},
        ];
        document.getElementById('statsGrid').innerHTML = cards.map(c=>(
            '<div class="stat-card"><div class="label">'+c.label+'</div>'+
            '<div class="value '+c.cls+'">'+c.value+'</div><div class="sub">'+c.sub+'</div></div>'
        )).join('');

        // 通用 tooltip 标题
        const ttTitle = ctx => ctx[0].label;

        // ===== 图表1：逐日收益率 =====
        const ctx1 = document.getElementById('chartReturns').getContext('2d');
        const barColors = returns.map(r=>r>=0?'rgba(63,185,80,0.75)':'rgba(248,81,73,0.75)');
        const barBorders = returns.map(r=>r>=0?'rgba(63,185,80,0.95)':'rgba(248,81,73,0.95)');
        const underSelectScatter = underSelectIdx.map(i=>({x:i, y:returns[i]}));

        new Chart(ctx1, {
            type:'bar',
            data:{labels:labels, datasets:[
                {label:'每日收益率 %', data:returns, backgroundColor:barColors, borderColor:barBorders,
                 borderWidth:0.3, borderRadius:1, barPercentage:0.85, categoryPercentage:0.9, order:1},
                {label:'10日移动平均', data:ma10, type:'line', borderColor:'#d29922',
                 backgroundColor:'rgba(210,153,34,0.08)', borderWidth:1.8, pointRadius:0, tension:0.35, fill:false, order:0},
                {label:'选股不足标记', data:underSelectScatter, type:'scatter', backgroundColor:'#f0883e',
                 borderColor:'#fff', borderWidth:1.2, pointRadius:6, pointStyle:'triangle', order:-1, showLine:false},
            ]},
            options:{
                responsive:true, maintainAspectRatio:false,
                interaction:{mode:'index', intersect:false},
                plugins:{
                    legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11},
                        filter:item=>!(item.datasetIndex===2 && underSelectScatter.length===0)}},
                    tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                        borderColor:'#58a6ff', borderWidth:1, padding:10,
                        callbacks:{
                            title:ttTitle,
                            label:c=>{
                                const i=c.dataIndex;
                                if(c.datasetIndex===0) return '收益: '+(returns[i]>=0?'+':'')+returns[i].toFixed(2)+'% | 选股: '+counts[i];
                                if(c.datasetIndex===1) return 'MA10: '+(ma10[i]>=0?'+':'')+ma10[i].toFixed(2)+'%';
                                if(c.datasetIndex===2) return '选股不足: '+counts[i]+' 只';
                                return '';
                            }
                        }
                    }
                },
                scales:{
                    x:{ticks:{color:'#8b949e', maxTicksLimit:30, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                    y:{title:{display:true, text:'收益率 (%)', color:'#8b949e'},
                       ticks:{color:'#8b949e', callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'}, grid:{color:'#30363d55'}}
                }
            }
        });

        // ===== 图表2：累计收益（复利 + 算术）=====
        const ctx2 = document.getElementById('chartCumulative').getContext('2d');
        new Chart(ctx2, {
            type:'line',
            data:{labels:labels, datasets:[
                {label:'复利累计(等权组合) %', data:cumCompound, borderColor:'#3fb950',
                 backgroundColor:'rgba(63,185,80,0.07)', borderWidth:2.2, pointRadius:0, tension:0.3, fill:true, order:0},
                {label:'算术累加(后端口径) %', data:cumArith, borderColor:'#58a6ff',
                 backgroundColor:'transparent', borderWidth:1.6, borderDash:[6,4], pointRadius:0, tension:0.3, fill:false, order:1},
            ]},
            options:{
                responsive:true, maintainAspectRatio:false,
                interaction:{mode:'index', intersect:false},
                plugins:{
                    legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                    tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                        borderColor:'#3fb950', borderWidth:1, padding:10,
                        callbacks:{title:ttTitle, label:c=>c.datasetIndex===0?
                            '复利累计: '+(cumCompound[c.dataIndex]>=0?'+':'')+cumCompound[c.dataIndex].toFixed(2)+'%':
                            '算术累加: '+(cumArith[c.dataIndex]>=0?'+':'')+cumArith[c.dataIndex].toFixed(2)+'%'}}
                },
                scales:{
                    x:{ticks:{color:'#8b949e', maxTicksLimit:30, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                    y:{title:{display:true, text:'累计收益率 (%)', color:'#8b949e'},
                       ticks:{color:'#8b949e', callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'}, grid:{color:'#30363d55'}}
                }
            },
            plugins:[{id:'zeroLineCum', beforeDraw(ch){
                const {ctx, scales}=ch; const y0=scales.y.getPixelForValue(0);
                ctx.save(); ctx.strokeStyle='rgba(230,237,243,0.45)'; ctx.lineWidth=1.3; ctx.setLineDash([8,4]);
                ctx.beginPath(); ctx.moveTo(scales.x.left,y0); ctx.lineTo(scales.x.right,y0); ctx.stroke(); ctx.setLineDash([]); ctx.restore();
            }}]
        });

        // ===== 图表3：回撤（基于复利资金曲线，已修正）=====
        const ctx3 = document.getElementById('chartDrawdown').getContext('2d');
        new Chart(ctx3, {
            type:'line',
            data:{labels:labels, datasets:[
                {label:'回撤 %', data:drawdowns, borderColor:'#f85149',
                 backgroundColor:'rgba(248,81,73,0.10)', borderWidth:1.8, pointRadius:0, tension:0.25, fill:true},
                {label:'最大回撤线', data:Array(N).fill(maxDrawdown), borderColor:'#f0883e',
                 backgroundColor:'transparent', borderWidth:1.2, borderDash:[8,5], pointRadius:0, tension:0, fill:false},
            ]},
            options:{
                responsive:true, maintainAspectRatio:false,
                interaction:{mode:'index', intersect:false},
                plugins:{
                    legend:{labels:{color:'#c9d1d9', usePointStyle:true, padding:16, font:{size:11}}},
                    tooltip:{backgroundColor:'#1c2129', titleColor:'#e6edf3', bodyColor:'#c9d1d9',
                        borderColor:'#f85149', borderWidth:1, padding:10,
                        callbacks:{title:ttTitle, label:c=>c.datasetIndex===0?
                            '回撤: '+drawdowns[c.dataIndex].toFixed(2)+'%':'最大回撤: '+maxDrawdown.toFixed(2)+'%'}}
                },
                scales:{
                    x:{ticks:{color:'#8b949e', maxTicksLimit:30, autoSkip:true, font:{size:9}}, grid:{color:'#30363d33'}},
                    y:{title:{display:true, text:'回撤 (%)', color:'#8b949e'},
                       ticks:{color:'#8b949e', callback:v=>v.toFixed(1)+'%'}, grid:{color:'#30363d55'}, max:0}
                }
            }
        });

        // ===== 控制台摘要 =====
        console.log('📊 ====== 模型赚钱效应分析（修正口径） ======');
        console.log('📌 有效交易日:', N);
        console.log('💰 复利累计收益:', fmtPct(totalCompound), '| 算术累加:', fmtPct(totalArith));
        console.log('✅ 胜率:', fmtNum(winRate,1)+'% ('+winCount+'赢/'+lossCount+'亏/'+flatCount+'平)');
        console.log('📈 平均每日收益:', fmtPct(avgReturn));
        console.log('🔻 最大回撤:', fmtNum(maxDrawdown,2)+'% (第'+(maxDrawdownIdx+1)+'日附近)');
        console.log('⚖️  盈亏比:', isFinite(profitLossRatio)?profitLossRatio.toFixed(2):'∞');
        if (totalCompound>0) console.log('🟢 结论: 复利口径下模型呈现正向赚钱效应');
        else if (totalCompound<0) console.log('🔴 结论: 复利口径下模型呈现负向效应，需优化');
        else console.log('⚪ 结论: 复利口径下盈亏平衡');
    })();
    </script>
</body>
</html>
'''


def find_latest_json(directory=OUT_RUN_DIR):
    """在 out_run/ 下找最新的 daily_stats_*.json"""
    pattern = os.path.join(directory, 'daily_stats_*.json')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_payload(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def render_html(payload):
    data = payload.get('daily', [])
    meta = payload.get('meta', {})
    html = (HTML_TEMPLATE
            .replace(_DATA_TOKEN, json.dumps(data, ensure_ascii=False))
            .replace(_META_TOKEN, json.dumps(meta, ensure_ascii=False)))
    return html


def main():
    parser = argparse.ArgumentParser(description='读取 out_run/ 每日统计 JSON 生成 HTML 可视化')
    parser.add_argument('json_path', nargs='?', default=None,
                        help='daily_stats_*.json 路径；省略则自动取 out_run/ 下最新一个')
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
        out_dir = os.path.dirname(os.path.abspath(json_path)) or OUT_RUN_DIR
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

// 市场指数看板前端逻辑
// 数据从 <script type="application/json" id="market-index-data"> 注入
(function () {
    const DATA_NODE = document.getElementById('market-index-data');
    let DATA = [];
    try {
        DATA = JSON.parse(DATA_NODE ? DATA_NODE.textContent : '[]') || [];
    } catch (e) {
        console.error('[market_index] 数据 JSON 解析失败:', e);
        DATA = [];
    }

    const daily = DATA.slice().sort((a, b) => a.yyyymmdd - b.yyyymmdd);
    const N = daily.length;
    const labels = daily.map(d => d.date);
    const avgChanges = daily.map(d => d.avg_change);
    const medianChanges = daily.map(d => d.median_change);
    const stockCounts = daily.map(d => d.stock_count);
    const upCounts = daily.map(d => d.up_count);
    const downCounts = daily.map(d => d.down_count);
    const flatCounts = daily.map(d => d.flat_count);

    // 格式化辅助：带正负号的百分比 / 数值 / 类别 class
    const fmtPct = v => (v >= 0 ? '+' : '') + (+v).toFixed(2) + '%';
    const fmtNum = (v, d = 2) => (+v).toFixed(d);
    const cls = (cond, negCond) => cond ? 'positive' : (negCond ? 'negative' : 'neutral');

    // ===== 复利净值 & 算术累计 =====
    const equity = [];
    const cumArith = [];
    let prod = 1, sumA = 0;
    for (let i = 0; i < N; i++) {
        prod *= (1 + avgChanges[i] / 100);
        sumA += avgChanges[i];
        equity.push(prod);
        cumArith.push(+sumA.toFixed(4));
    }

    // ===== 10日移动平均 =====
    const ma10 = [];
    for (let i = 0; i < N; i++) {
        const s = Math.max(0, i - 9);
        const slice = avgChanges.slice(s, i + 1);
        ma10.push(+(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(4));
    }

    // ===== 回撤 =====
    const drawdowns = [];
    let peakEq = equity.length ? equity[0] : 1;
    for (let i = 0; i < N; i++) {
        if (equity[i] > peakEq) peakEq = equity[i];
        const dd = peakEq > 0 ? (equity[i] - peakEq) / peakEq * 100 : 0;
        drawdowns.push(+dd.toFixed(4));
    }
    const maxDrawdown = N ? Math.min.apply(null, drawdowns) : 0;

    // ===== 涨跌比 =====
    const upDownRatio = daily.map(d =>
        d.down_count > 0
            ? +(d.up_count / d.down_count).toFixed(2)
            : (d.up_count > 0 ? 99 : 0)
    );

    // ===== 统计量 =====
    const winCount = avgChanges.filter(r => r > 0).length;
    const lossCount = avgChanges.filter(r => r < 0).length;
    const winRate = N ? winCount / N * 100 : 0;
    const totalArith = N ? cumArith[N - 1] : 0;
    const totalCompound = N ? +((equity[N - 1] - 1) * 100).toFixed(4) : 0;
    const avgReturn = N ? avgChanges.reduce((a, b) => a + b, 0) / N : 0;
    const maxSingleUp = N ? Math.max.apply(null, avgChanges) : 0;
    const maxSingleDown = N ? Math.min.apply(null, avgChanges) : 0;
    const variance = N ? avgChanges.reduce((s, r) => s + (r - avgReturn) * (r - avgReturn), 0) / N : 0;
    const stdDev = Math.sqrt(variance);
    const sharpeApprox = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(N) : 0;
    const avgStockCount = N ? stockCounts.reduce((a, b) => a + b, 0) / N : 0;

    // ===== Meta 行 =====
    const startDate = N ? daily[0].date : '?';
    const endDate = N ? daily[N - 1].date : '?';
    document.getElementById('metaLine').textContent =
        '区间: ' + startDate + ' ~ ' + endDate + '  |  平均覆盖: ' + avgStockCount.toFixed(0) + ' 只股票/日';
    document.getElementById('totalDaysFoot').textContent = N;

    // ===== 时间范围快捷按钮 =====
    const rangeDiv = document.getElementById('rangeControls');
    const ranges = [
        { label: '全部', months: 0 },
        { label: '近1年', months: 12 },
        { label: '近6月', months: 6 },
        { label: '近3月', months: 3 },
    ];
    // 存储原始数据供范围切换
    window._fullData = {
        daily, N, labels, avgChanges, medianChanges,
        stockCounts, upCounts, downCounts, flatCounts,
        equity, cumArith, ma10, drawdowns, upDownRatio,
    };

    ranges.forEach((r, idx) => {
        const btn = document.createElement('button');
        btn.textContent = r.label;
        if (idx === 0) btn.classList.add('active');
        btn.addEventListener('click', function () {
            rangeDiv.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyRange(r.months);
        });
        rangeDiv.appendChild(btn);
    });

    function applyRange(months) {
        // months=0 表示全部
        const F = window._fullData;
        let startIdx = 0;
        if (months > 0 && F.N > 0) {
            const lastDate = F.daily[F.N - 1].yyyymmdd;
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

        const wc = changes.filter(r => r > 0).length;
        const lc = changes.filter(r => r < 0).length;
        const wr = wc / n * 100;
        const ta = cumA[cumA.length - 1];
        const tc = +((eq[eq.length - 1] - 1) * 100).toFixed(4);
        const ar = changes.reduce((a, b) => a + b, 0) / n;
        const maxUp = Math.max.apply(null, changes);
        const maxDown = Math.min.apply(null, changes);
        const v = changes.reduce((s, r) => s + (r - ar) * (r - ar), 0) / n;
        const sd = Math.sqrt(v);
        const sh = sd > 0 ? (ar / sd) * Math.sqrt(n) : 0;
        const mdd = Math.min.apply(null, dd);
        const asc = counts.reduce((a, b) => a + b, 0) / n;

        const cards = [
            { label: '交易日', value: String(n), cls: 'neutral', sub: '有效交易天数' },
            { label: '复利累计', value: fmtPct(tc), cls: cls(tc >= 0, true), sub: '等权组合复利' },
            { label: '算术累计', value: fmtPct(ta), cls: cls(ta >= 0, true), sub: 'Σ 每日平均' },
            { label: '上涨日占比', value: fmtNum(wr, 1) + '%', cls: cls(wr >= 50, true), sub: wc + '涨/' + lc + '跌' },
            { label: '日均涨跌幅', value: fmtPct(ar), cls: cls(ar >= 0, true), sub: '期望值' },
            { label: '最大单日涨幅', value: '+' + fmtNum(maxUp, 2) + '%', cls: 'highlight', sub: '最佳交易日' },
            { label: '最大单日跌幅', value: fmtNum(maxDown, 2) + '%', cls: 'negative', sub: '最差交易日' },
            { label: '最大回撤', value: fmtNum(mdd, 2) + '%', cls: 'negative', sub: '基于复利净值' },
            { label: '夏普近似', value: fmtNum(sh, 2), cls: sh >= 1 ? 'positive' : 'neutral', sub: '风险调整收益' },
            { label: '日均股票数', value: fmtNum(asc, 0), cls: 'neutral', sub: '覆盖股票只数' },
        ];
        document.getElementById('statsGrid').innerHTML = cards.map(c =>
            '<div class="stat-card"><div class="label">' + c.label + '</div>' +
            '<div class="value ' + c.cls + '">' + c.value + '</div><div class="sub">' + c.sub + '</div></div>'
        ).join('');
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
        const barColors = sc.map(r => r >= 0 ? 'rgba(63,185,80,0.75)' : 'rgba(248,81,73,0.75)');
        const barBorders = sc.map(r => r >= 0 ? 'rgba(63,185,80,0.95)' : 'rgba(248,81,73,0.95)');

        if (chartChange) chartChange.destroy();
        chartChange = new Chart(document.getElementById('chartChange').getContext('2d'), {
            type: 'bar',
            data: {
                labels: sl,
                datasets: [
                    {
                        label: '平均涨跌幅 %', data: sc,
                        backgroundColor: barColors, borderColor: barBorders,
                        borderWidth: 0.3, borderRadius: 1,
                        barPercentage: 0.85, categoryPercentage: 0.9, order: 1,
                    },
                    {
                        label: '10日移动平均', data: sma, type: 'line',
                        borderColor: '#d29922',
                        backgroundColor: 'rgba(210,153,34,0.08)',
                        borderWidth: 1.8, pointRadius: 0, tension: 0.35, fill: false, order: 0,
                    },
                    {
                        label: '中位数涨跌幅', data: sm, type: 'line',
                        borderColor: '#a371f7',
                        backgroundColor: 'transparent',
                        borderWidth: 1.2, borderDash: [4, 3], pointRadius: 0, tension: 0.35, fill: false, order: -1,
                    },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#c9d1d9', usePointStyle: true, padding: 16, font: { size: 11 } } },
                    tooltip: {
                        backgroundColor: '#1c2129', titleColor: '#e6edf3', bodyColor: '#c9d1d9',
                        borderColor: '#58a6ff', borderWidth: 1, padding: 10,
                        callbacks: {
                            label: c => {
                                const i = c.dataIndex;
                                if (c.datasetIndex === 0) return '均值: ' + fmtPct(sc[i]) + ' | 股票: ' + ssc[i];
                                if (c.datasetIndex === 1) return 'MA10: ' + fmtPct(sma[i]);
                                if (c.datasetIndex === 2) return '中位数: ' + fmtPct(sm[i]);
                                return '';
                            },
                        },
                    },
                },
                scales: {
                    x: { ticks: { color: '#8b949e', maxTicksLimit: 40, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                    y: {
                        title: { display: true, text: '涨跌幅 (%)', color: '#8b949e' },
                        ticks: { color: '#8b949e', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' },
                        grid: { color: '#30363d55' },
                    },
                },
            },
        });

        // ── 图2: 累计指数 ──
        const cumCompound = seq.map(e => +((e - 1) * 100).toFixed(4));
        if (chartCumulative) chartCumulative.destroy();
        chartCumulative = new Chart(document.getElementById('chartCumulative').getContext('2d'), {
            type: 'line',
            data: {
                labels: sl,
                datasets: [
                    {
                        label: '复利净值', data: seq, borderColor: '#3fb950',
                        backgroundColor: 'rgba(63,185,80,0.07)', borderWidth: 2.2,
                        pointRadius: 0, tension: 0.3, fill: true,
                        yAxisID: 'y', order: 0,
                    },
                    {
                        label: '算术累计 %', data: sca, borderColor: '#58a6ff',
                        backgroundColor: 'transparent', borderWidth: 1.6, borderDash: [6, 4],
                        pointRadius: 0, tension: 0.3, fill: false,
                        yAxisID: 'y1', order: 1,
                    },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#c9d1d9', usePointStyle: true, padding: 16, font: { size: 11 } } },
                    tooltip: {
                        backgroundColor: '#1c2129', titleColor: '#e6edf3', bodyColor: '#c9d1d9',
                        borderColor: '#3fb950', borderWidth: 1, padding: 10,
                        callbacks: {
                            label: c => c.datasetIndex === 0
                                ? '净值: ' + seq[c.dataIndex].toFixed(4) + ' (' + fmtPct(cumCompound[c.dataIndex]) + ')'
                                : '算术累计: ' + fmtPct(sca[c.dataIndex]),
                        },
                    },
                },
                scales: {
                    x: { ticks: { color: '#8b949e', maxTicksLimit: 30, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                    y: {
                        type: 'linear', position: 'left',
                        title: { display: true, text: '复利净值', color: '#3fb950' },
                        ticks: { color: '#3fb950', callback: v => v.toFixed(2) },
                        grid: { color: '#30363d55' },
                    },
                    y1: {
                        type: 'linear', position: 'right',
                        title: { display: true, text: '算术累计 %', color: '#58a6ff' },
                        ticks: { color: '#58a6ff', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
            plugins: [{
                id: 'baseLine',
                beforeDraw(ch) {
                    const { ctx, scales } = ch;
                    const y0 = scales.y.getPixelForValue(1);
                    ctx.save();
                    ctx.strokeStyle = 'rgba(230,237,243,0.45)';
                    ctx.lineWidth = 1.3;
                    ctx.setLineDash([8, 4]);
                    ctx.beginPath();
                    ctx.moveTo(scales.x.left, y0);
                    ctx.lineTo(scales.x.right, y0);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.restore();
                },
            }],
        });

        // ── 图3: 涨跌家数 ──
        if (chartUpDown) chartUpDown.destroy();
        chartUpDown = new Chart(document.getElementById('chartUpDown').getContext('2d'), {
            type: 'bar',
            data: {
                labels: sl,
                datasets: [
                    { label: '上涨', data: su, backgroundColor: 'rgba(63,185,80,0.7)', borderColor: 'rgba(63,185,80,0.9)', borderWidth: 0.3, borderRadius: 1, stack: 'stack0', order: 2 },
                    { label: '下跌', data: sd2.map(v => -v), backgroundColor: 'rgba(248,81,73,0.7)', borderColor: 'rgba(248,81,73,0.9)', borderWidth: 0.3, borderRadius: 1, stack: 'stack0', order: 2 },
                    { label: '平盘', data: sf, backgroundColor: 'rgba(139,148,158,0.5)', borderColor: 'rgba(139,148,158,0.7)', borderWidth: 0.3, borderRadius: 1, stack: 'stack1', order: 3 },
                    {
                        label: '涨跌比', data: sr, type: 'line', borderColor: '#d29922',
                        backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false,
                        yAxisID: 'y1', order: 0,
                    },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#c9d1d9', usePointStyle: true, padding: 16, font: { size: 11 } } },
                    tooltip: {
                        backgroundColor: '#1c2129', titleColor: '#e6edf3', bodyColor: '#c9d1d9',
                        borderColor: '#d29922', borderWidth: 1, padding: 10,
                        callbacks: {
                            label: c => {
                                const i = c.dataIndex;
                                if (c.datasetIndex === 0) return '上涨: ' + su[i];
                                if (c.datasetIndex === 1) return '下跌: ' + sd2[i];
                                if (c.datasetIndex === 2) return '平盘: ' + sf[i];
                                if (c.datasetIndex === 3) return '涨跌比: ' + sr[i].toFixed(2);
                                return '';
                            },
                        },
                    },
                },
                scales: {
                    x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 40, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                    y: {
                        stacked: true,
                        title: { display: true, text: '家数', color: '#8b949e' },
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d55' },
                    },
                    y1: {
                        type: 'linear', position: 'right',
                        title: { display: true, text: '涨跌比', color: '#d29922' },
                        ticks: { color: '#d29922' }, grid: { drawOnChartArea: false }, min: 0,
                    },
                },
            },
        });

        // ── 图4: 回撤 ──
        if (chartDrawdown) chartDrawdown.destroy();
        chartDrawdown = new Chart(document.getElementById('chartDrawdown').getContext('2d'), {
            type: 'line',
            data: {
                labels: sl,
                datasets: [
                    { label: '回撤 %', data: sdd, borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.10)', borderWidth: 1.8, pointRadius: 0, tension: 0.25, fill: true },
                    { label: '最大回撤线', data: Array(sdd.length).fill(mdd), borderColor: '#f0883e', backgroundColor: 'transparent', borderWidth: 1.2, borderDash: [8, 5], pointRadius: 0, tension: 0, fill: false },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#c9d1d9', usePointStyle: true, padding: 16, font: { size: 11 } } },
                    tooltip: {
                        backgroundColor: '#1c2129', titleColor: '#e6edf3', bodyColor: '#c9d1d9',
                        borderColor: '#f85149', borderWidth: 1, padding: 10,
                        callbacks: {
                            label: c => c.datasetIndex === 0
                                ? '回撤: ' + sdd[c.dataIndex].toFixed(2) + '%'
                                : '最大回撤: ' + mdd.toFixed(2) + '%',
                        },
                    },
                },
                scales: {
                    x: { ticks: { color: '#8b949e', maxTicksLimit: 30, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                    y: {
                        title: { display: true, text: '回撤 (%)', color: '#8b949e' },
                        ticks: { color: '#8b949e', callback: v => v.toFixed(1) + '%' },
                        grid: { color: '#30363d55' }, max: 0,
                    },
                },
            },
        });
    }

    // 初始渲染
    updateCharts(0);
})();

// 模型赚钱效应看板前端逻辑
// 数据从 <script type="application/json" id="daily-data">…</script> 与
// <script type="application/json" id="daily-meta">…</script> 注入
(function () {
    function readJSON(id, fallback) {
        const node = document.getElementById(id);
        if (!node) return fallback;
        try {
            const v = JSON.parse(node.textContent);
            return v == null ? fallback : v;
        } catch (e) {
            console.error('[visualize_daily] ' + id + ' 解析失败:', e);
            return fallback;
        }
    }
    const DATA = readJSON('daily-data', []);
    const META = readJSON('daily-meta', {});

    // ===== 基础序列 =====
    const daily = (DATA || []).slice().sort((a, b) => a.yyyymmdd - b.yyyymmdd);
    const N = daily.length;
    const labels = daily.map(d => d.date);
    const returns = daily.map(d => d.return_pct);     // 每日收益率 %
    const counts = daily.map(d => d.count);
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
    for (let i = 0; i < N; i++) {
        prod *= (1 + returns[i] / 100);
        sumA += returns[i];
        equity.push(prod);
        cumArith.push(+sumA.toFixed(4));
        cumCompound.push(+((prod - 1) * 100).toFixed(4));
    }

    // ===== 10日移动平均 =====
    const ma10 = [];
    for (let i = 0; i < N; i++) {
        const s = Math.max(0, i - 9);
        const slice = returns.slice(s, i + 1);
        ma10.push(+(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(4));
    }

    // ===== 回撤：在复利资金曲线 equity 上计算标准相对回撤 =====
    // 修正点：原文件在"累计百分比"序列上做 (cur-peak)/peak，peak≈0 时失真。
    // 这里 equity>=0、peakEquity>=1，除法恒良定义。
    const drawdowns = [];
    let peakEq = equity.length ? equity[0] : 1;
    for (let i = 0; i < N; i++) {
        if (equity[i] > peakEq) peakEq = equity[i];
        const dd = peakEq > 0 ? (equity[i] - peakEq) / peakEq * 100 : 0;
        drawdowns.push(+dd.toFixed(4));
    }
    const maxDrawdown = N ? Math.min.apply(null, drawdowns) : 0;
    const maxDrawdownIdx = drawdowns.indexOf(maxDrawdown);

    // ===== 统计量 =====
    const winCount = returns.filter(r => r > 0).length;
    const lossCount = returns.filter(r => r < 0).length;
    const flatCount = returns.filter(r => r === 0).length;
    const winRate = N ? winCount / N * 100 : 0;
    const totalArith = N ? cumArith[N - 1] : 0;
    const totalCompound = N ? cumCompound[N - 1] : 0;
    const avgReturn = N ? returns.reduce((a, b) => a + b, 0) / N : 0;
    const maxSingleWin = N ? Math.max.apply(null, returns) : 0;
    const maxSingleLoss = N ? Math.min.apply(null, returns) : 0;
    const variance = N ? returns.reduce((s, r) => s + (r - avgReturn) * (r - avgReturn), 0) / N : 0;
    const stdDev = Math.sqrt(variance);
    const sharpeApprox = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(N) : 0;
    const avgWin = winCount > 0 ? returns.filter(r => r > 0).reduce((a, b) => a + b, 0) / winCount : 0;
    const avgLoss = lossCount > 0 ? Math.abs(returns.filter(r => r < 0).reduce((a, b) => a + b, 0) / lossCount) : 0;
    const profitLossRatio = avgLoss > 0 ? avgWin / avgLoss : (avgWin > 0 ? Infinity : 0);
    const avgCount = N ? counts.reduce((a, b) => a + b, 0) / N : 0;
    const underSelectIdx = counts.map((c, i) => c < maxSelect ? i : -1).filter(i => i >= 0);

    // ===== 顶部 meta 行 =====
    document.getElementById('metaLine').textContent =
        '模型: ' + (META.model || '?') + '  |  区间: ' + (META.start_date || '?') + ' ~ ' + (META.end_date || '?') +
        (META.begin_date ? '  |  --begin: ' + META.begin_date : '') +
        '  |  阈值: ' + (META.threshold != null ? (+META.threshold).toFixed(6) : '?');
    document.getElementById('totalDaysFoot').textContent = N;

    // ===== 统计卡片 =====
    const fmtPct = v => (v >= 0 ? '+' : '') + (+v).toFixed(2) + '%';
    const fmtNum = (v, d = 2) => (+v).toFixed(d);
    const cls = (cond, negCond) => cond ? 'positive' : (negCond ? 'negative' : 'neutral');
    const cards = [
        { label: '有效交易日', value: String(N), cls: 'neutral', sub: 'count>0 的日子' },
        { label: '复利累计收益', value: fmtPct(totalCompound), cls: cls(totalCompound >= 0, true), sub: '等权组合复利' },
        { label: '算术累计收益', value: fmtPct(totalArith), cls: cls(totalArith >= 0, true), sub: 'Σ 每日(后端口径)' },
        { label: '胜率', value: fmtNum(winRate, 1) + '%', cls: cls(winRate >= 50, true), sub: winCount + '赢/' + lossCount + '亏/' + flatCount + '平' },
        { label: '平均每日收益', value: fmtPct(avgReturn), cls: cls(avgReturn >= 0, true), sub: '期望值' },
        { label: '最大单日盈利', value: '+' + fmtNum(maxSingleWin, 1) + '%', cls: 'highlight', sub: '最佳交易日' },
        { label: '最大单日亏损', value: fmtNum(maxSingleLoss, 1) + '%', cls: 'negative', sub: '最差交易日' },
        { label: '最大回撤', value: fmtNum(maxDrawdown, 2) + '%', cls: 'negative', sub: maxDrawdownIdx >= 0 ? ('第' + (maxDrawdownIdx + 1) + '日附近') : '-' },
        { label: '盈亏比', value: isFinite(profitLossRatio) ? fmtNum(profitLossRatio, 2) : '∞', cls: profitLossRatio >= 2 ? 'positive' : 'neutral', sub: 'avgWin/avgLoss' },
        { label: '夏普近似', value: fmtNum(sharpeApprox, 2), cls: sharpeApprox >= 1 ? 'positive' : 'neutral', sub: '风险调整收益' },
        { label: '平均选股数', value: fmtNum(avgCount, 1), cls: 'neutral', sub: '每日入选只数' },
    ];
    document.getElementById('statsGrid').innerHTML = cards.map(c =>
        '<div class="stat-card"><div class="label">' + c.label + '</div>' +
        '<div class="value ' + c.cls + '">' + c.value + '</div><div class="sub">' + c.sub + '</div></div>'
    ).join('');

    // 通用 tooltip 标题
    const ttTitle = ctx => ctx[0].label;

    // ===== 图表1：逐日收益率 =====
    const ctx1 = document.getElementById('chartReturns').getContext('2d');
    const barColors = returns.map(r => r >= 0 ? 'rgba(63,185,80,0.75)' : 'rgba(248,81,73,0.75)');
    const barBorders = returns.map(r => r >= 0 ? 'rgba(63,185,80,0.95)' : 'rgba(248,81,73,0.95)');
    const underSelectScatter = underSelectIdx.map(i => ({ x: i, y: returns[i] }));

    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '每日收益率 %', data: returns,
                    backgroundColor: barColors, borderColor: barBorders,
                    borderWidth: 0.3, borderRadius: 1,
                    barPercentage: 0.85, categoryPercentage: 0.9, order: 1,
                },
                {
                    label: '10日移动平均', data: ma10, type: 'line',
                    borderColor: '#d29922',
                    backgroundColor: 'rgba(210,153,34,0.08)',
                    borderWidth: 1.8, pointRadius: 0, tension: 0.35, fill: false, order: 0,
                },
                {
                    label: '选股不足标记', data: underSelectScatter, type: 'scatter',
                    backgroundColor: '#f0883e',
                    borderColor: '#fff', borderWidth: 1.2,
                    pointRadius: 6, pointStyle: 'triangle', order: -1, showLine: false,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color: '#c9d1d9', usePointStyle: true, padding: 16, font: { size: 11 },
                        filter: item => !(item.datasetIndex === 2 && underSelectScatter.length === 0),
                    },
                },
                tooltip: {
                    backgroundColor: '#1c2129', titleColor: '#e6edf3', bodyColor: '#c9d1d9',
                    borderColor: '#58a6ff', borderWidth: 1, padding: 10,
                    callbacks: {
                        title: ttTitle,
                        label: c => {
                            const i = c.dataIndex;
                            if (c.datasetIndex === 0) return '收益: ' + fmtPct(returns[i]) + ' | 选股: ' + counts[i];
                            if (c.datasetIndex === 1) return 'MA10: ' + fmtPct(ma10[i]);
                            if (c.datasetIndex === 2) return '选股不足: ' + counts[i] + ' 只';
                            return '';
                        },
                    },
                },
            },
            scales: {
                x: { ticks: { color: '#8b949e', maxTicksLimit: 30, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                y: {
                    title: { display: true, text: '收益率 (%)', color: '#8b949e' },
                    ticks: { color: '#8b949e', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' },
                    grid: { color: '#30363d55' },
                },
            },
        },
    });

    // ===== 图表2：累计收益（复利 + 算术）=====
    const ctx2 = document.getElementById('chartCumulative').getContext('2d');
    new Chart(ctx2, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '复利累计(等权组合) %', data: cumCompound, borderColor: '#3fb950',
                    backgroundColor: 'rgba(63,185,80,0.07)', borderWidth: 2.2,
                    pointRadius: 0, tension: 0.3, fill: true, order: 0,
                },
                {
                    label: '算术累加(后端口径) %', data: cumArith, borderColor: '#58a6ff',
                    backgroundColor: 'transparent', borderWidth: 1.6, borderDash: [6, 4],
                    pointRadius: 0, tension: 0.3, fill: false, order: 1,
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
                        title: ttTitle,
                        label: c => c.datasetIndex === 0
                            ? '复利累计: ' + fmtPct(cumCompound[c.dataIndex])
                            : '算术累加: ' + fmtPct(cumArith[c.dataIndex]),
                    },
                },
            },
            scales: {
                x: { ticks: { color: '#8b949e', maxTicksLimit: 30, autoSkip: true, font: { size: 9 } }, grid: { color: '#30363d33' } },
                y: {
                    title: { display: true, text: '累计收益率 (%)', color: '#8b949e' },
                    ticks: { color: '#8b949e', callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%' },
                    grid: { color: '#30363d55' },
                },
            },
        },
        plugins: [{
            id: 'zeroLineCum',
            beforeDraw(ch) {
                const { ctx, scales } = ch;
                const y0 = scales.y.getPixelForValue(0);
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

    // ===== 图表3：回撤（基于复利资金曲线，已修正）=====
    const ctx3 = document.getElementById('chartDrawdown').getContext('2d');
    new Chart(ctx3, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '回撤 %', data: drawdowns, borderColor: '#f85149',
                    backgroundColor: 'rgba(248,81,73,0.10)', borderWidth: 1.8,
                    pointRadius: 0, tension: 0.25, fill: true,
                },
                {
                    label: '最大回撤线', data: Array(N).fill(maxDrawdown),
                    borderColor: '#f0883e',
                    backgroundColor: 'transparent', borderWidth: 1.2, borderDash: [8, 5],
                    pointRadius: 0, tension: 0, fill: false,
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
                    borderColor: '#f85149', borderWidth: 1, padding: 10,
                    callbacks: {
                        title: ttTitle,
                        label: c => c.datasetIndex === 0
                            ? '回撤: ' + drawdowns[c.dataIndex].toFixed(2) + '%'
                            : '最大回撤: ' + maxDrawdown.toFixed(2) + '%',
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

    // ===== 控制台摘要 =====
    console.log('📊 ====== 模型赚钱效应分析（修正口径） ======');
    console.log('📌 有效交易日:', N);
    console.log('💰 复利累计收益:', fmtPct(totalCompound), '| 算术累加:', fmtPct(totalArith));
    console.log('✅ 胜率:', fmtNum(winRate, 1) + '% (' + winCount + '赢/' + lossCount + '亏/' + flatCount + '平)');
    console.log('📈 平均每日收益:', fmtPct(avgReturn));
    console.log('🔻 最大回撤:', fmtNum(maxDrawdown, 2) + '% (第' + (maxDrawdownIdx + 1) + '日附近)');
    console.log('⚖️  盈亏比:', isFinite(profitLossRatio) ? profitLossRatio.toFixed(2) : '∞');
    if (totalCompound > 0) console.log('🟢 结论: 复利口径下模型呈现正向赚钱效应');
    else if (totalCompound < 0) console.log('🔴 结论: 复利口径下模型呈现负向效应，需优化');
    else console.log('⚪ 结论: 复利口径下盈亏平衡');
})();
#!/usr/bin/env python3
"""
EquiNet 训练实时监控服务（含系统资源监控）

用法:
    python tools/train_monitor.py --port 8080

浏览器打开 http://<服务器IP>:8080/

默认【自动跟随】训练：目标日志取 train_logs/current_run.txt 指向的那份（没有就取
最新的 .log），逐步指标按日志名的时间戳去 out/ 下匹配同一时刻的 step_metrics.csv。
所以换一次训练不用改监控、也不用改命令行；页面右上角还能直接切到历史运行。
需要钉死看某一份文件时才用 --log / --steps / --compare 覆盖。

图表用 Apache ECharts 5.5.1，本地托管于 <仓库根>/static/echarts.min.js，
不依赖 CDN。该文件属于第三方库、不入仓库，缺失时页面会提示；可用 --static 指定目录。

系统资源由后台线程每 2.5 秒采样一次（CPU / 内存 / GPU / 显存 / 磁盘），
图上只显示最近 5 分钟 —— 任务管理器式的滑动窗口。
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
import glob
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# ECharts 等第三方静态资源不入库；默认从仓库根目录的 static/ 读取，
# 可用 --static 指定其它目录（需包含 echarts.min.js）。
STATIC_DIR = os.path.join(REPO_ROOT, 'static')

# 训练进程把日志写到仓库【同级】的 train_logs/，并用 current_run.txt 指认当前
# 那一次；out/ 下每次运行一个时间戳目录。监控默认就从这两处自动发现目标，
# 不需要为每次训练重新拼命令行参数。
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(REPO_ROOT), 'train_logs')
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, 'out')

SAMPLE_INTERVAL = 2.5      # 资源采样间隔（秒），与前端轮询一致
RES_WINDOW_SEC = 300       # 图上显示最近多少秒的资源历史
RES_HIST_MAX = 4000        # 环形缓冲上限（约 2.7 小时）

# ========================= 系统资源采集 =========================

_RES_HIST = collections.deque(maxlen=RES_HIST_MAX)
_RES_LOCK = threading.Lock()
_cpu_prev = None


def _read_cpu_stat():
    """读 /proc/stat 第一行，返回 (idle_jiffies, total_jiffies)。"""
    with open('/proc/stat', 'r') as f:
        parts = [int(x) for x in f.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)   # idle + iowait
    return idle, sum(parts)


def _cpu_percent():
    """用两次采样之差算 CPU 占用率。第一次返回 None。"""
    global _cpu_prev
    idle, total = _read_cpu_stat()
    pct = None
    if _cpu_prev is not None:
        pi, pt = _cpu_prev
        dt = total - pt
        if dt > 0:
            pct = max(0.0, min(100.0, 100.0 * (1.0 - (idle - pi) / dt)))
    _cpu_prev = (idle, total)
    return pct


def _mem_info():
    info = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            k, _, v = line.partition(':')
            info[k.strip()] = int(v.split()[0])      # kB
    total = info.get('MemTotal', 0) / 1024 / 1024
    avail = info.get('MemAvailable', 0) / 1024 / 1024
    used = total - avail
    return {'total_gb': round(total, 1), 'used_gb': round(used, 1),
            'avail_gb': round(avail, 1),
            'pct': round(100.0 * used / total, 1) if total else None}


def _gpu_info():
    """nvidia-smi 查询。失败时返回空 dict，不影响其它指标。"""
    out = {}
    try:
        q = ('utilization.gpu,memory.used,memory.total,temperature.gpu,'
             'power.draw,power.limit,fan.speed')
        r = subprocess.run(
            ['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=6)
        if r.returncode == 0 and r.stdout.strip():
            v = [x.strip() for x in r.stdout.strip().split('\n')[0].split(',')]

            def f(i):
                try:
                    return float(v[i])
                except (ValueError, IndexError):
                    return None
            used, total = f(1), f(2)
            out = {'util': f(0), 'mem_used_mb': used, 'mem_total_mb': total,
                   'mem_pct': round(100.0 * used / total, 1) if used and total else None,
                   'temp_c': f(3), 'power_w': f(4), 'power_limit_w': f(5),
                   'fan_pct': f(6)}
            out['mem_used_gb'] = round(used / 1024, 2) if used else None
            out['mem_total_gb'] = round(total / 1024, 2) if total else None
    except Exception:
        pass
    # GPU 上的进程
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=6)
        procs = []
        if r.returncode == 0:
            for ln in r.stdout.strip().split('\n'):
                if not ln.strip():
                    continue
                a = [x.strip() for x in ln.split(',')]
                if len(a) >= 2:
                    try:
                        procs.append({'pid': int(a[0]), 'mem_mb': float(a[1])})
                    except ValueError:
                        pass
        out['procs'] = procs
    except Exception:
        out['procs'] = []
    return out


def _disk_info(path='/home'):
    # 用 1024^3 换算成 GiB —— 与 `df -h` 的口径一致。
    # 之前用 1e9（十进制 GB）会显示成 1006/757 GB，而 df 显示 937/706，
    # 数值本身没错但单位混用会让人以为采错了。
    G = 1024 ** 3
    try:
        u = shutil.disk_usage(path)
        # 百分比用 used/(used+free)，与 `df` 的口径一致。
        # 用 used/total 会偏小（文件系统有保留块），df 显示 80% 而面板显示 75%，
        # 会让人以为采集错了。
        denom = u.used + u.free
        return {'total_g': round(u.total / G, 1),
                'used_g': round(u.used / G, 1),
                'free_g': round(u.free / G, 1),
                'pct': round(100.0 * u.used / denom, 1) if denom else None}
    except Exception:
        return {}


def collect_system():
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = (None, None, None)
    cpu = _cpu_percent()
    mem = _mem_info()
    gpu = _gpu_info()
    return {
        't': time.time(),
        'cpu_pct': cpu,
        'cpu_count': os.cpu_count(),
        'load1': round(load[0], 2) if load[0] is not None else None,
        'load5': round(load[1], 2) if load[1] is not None else None,
        'load15': round(load[2], 2) if load[2] is not None else None,
        'mem': mem,
        'mem_pct': mem.get('pct'),
        'gpu': gpu,
        'gpu_util': gpu.get('util'),
        'vram_pct': gpu.get('mem_pct'),
        'disk': _disk_info(),
    }


def _sampler_loop(stop_evt):
    _cpu_percent()                     # 丢弃第一次（无前值）
    while not stop_evt.wait(SAMPLE_INTERVAL):
        try:
            s = collect_system()
            with _RES_LOCK:
                _RES_HIST.append(s)
        except Exception:
            pass


def res_history(window_sec=RES_WINDOW_SEC):
    now = time.time()
    with _RES_LOCK:
        pts = [s for s in _RES_HIST if now - s['t'] <= window_sec]
    return pts


# ========================= 运行目标发现 =========================
#
# 监控不该跟某一次实验绑定。训练侧只约定两件事：
#   1. 日志落在 runs_dir 下，命名为 <名称>_<YYYYMMDD>_<HHMMSS>.log
#      （train.py 的启动脚本已经这样做），并把当前那次写进 current_run.txt；
#   2. 每步指标写在 out/<YYYY-MM-DD_HHMMSS>/step_metrics.csv。
# 监控据此自己找目标：默认跟 current_run.txt，找不到就取最新的日志；
# 逐步指标按日志名里的时间戳去找同一时刻创建的 out 目录。
# 于是换一次训练不需要改监控，也不需要在页面上改任何东西。

RE_LOG_STAMP = re.compile(r'(\d{8})_(\d{6})')


def _parse_ts(text, fmt):
    try:
        return time.mktime(time.strptime(text, fmt))
    except (ValueError, OverflowError):
        return None


def log_timestamp(path):
    """从日志文件名里取出 YYYYMMDD_HHMMSS 的时间戳（秒）。取不到返回 None。"""
    m = RE_LOG_STAMP.search(os.path.basename(path or ''))
    return _parse_ts(m.group(1) + m.group(2), '%Y%m%d%H%M%S') if m else None


def read_current_run(runs_dir):
    """读 runs_dir/current_run.txt，返回其中记录的日志路径（不存在则 None）。"""
    marker = os.path.join(runs_dir, 'current_run.txt')
    try:
        with open(marker, encoding='utf-8', errors='replace') as f:
            p = f.read().strip()
    except OSError:
        return None
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(runs_dir, p)
    return p if os.path.isfile(p) else None


# train_logs/ 里除了训练日志，还混着评估守护、监控、数据维护的日志，它们的
# 文件名同样带时间戳，日志开头打印的配置块也很像。真正只有训练日志会出现
# 轮次标题 / 训练进度行，按这个判别（分块扫描，通常在第一个块就命中）。
_LOG_MARK = re.compile(r'Epoch\s+\d+/\d+\s*·|训练进度:')
_LOG_VERDICT = {}


def looks_like_training_log(path, sniff=2_000_000, chunk=262144):
    """判断这个 .log 是不是 train.py 的训练日志（按内容，带缓存）。"""
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (st.st_mtime_ns, min(st.st_size, sniff))
    hit = _LOG_VERDICT.get(path)
    if hit and hit[0] == key:
        return hit[1]
    ok = False
    try:
        with open(path, 'rb') as f:
            done = 0
            while done < sniff:
                buf = f.read(chunk)
                if not buf:
                    break
                done += len(buf)
                if _LOG_MARK.search(buf.decode('utf-8', 'replace')):
                    ok = True
                    break
    except OSError:
        ok = False
    _LOG_VERDICT[path] = (key, ok)
    return ok


def list_runs(runs_dir, limit=40):
    """列出 runs_dir 下的训练日志，最新的在前。

    先按命名约定粗筛（<名称>_<YYYYMMDD>_<HHMMSS>.log），再用内容确认确实是
    train.py 的训练日志；软链（例如 current.log）按真实文件去重。
    """
    if not runs_dir or not os.path.isdir(runs_dir):
        return []
    seen, runs = set(), []
    for p in glob.glob(os.path.join(runs_dir, '*.log')):
        if not RE_LOG_STAMP.search(os.path.basename(p)):
            continue
        try:
            real = os.path.realpath(p)
            st = os.stat(real)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        runs.append({'name': os.path.basename(real), 'path': real,
                     'mtime': st.st_mtime, 'size': st.st_size})
    runs.sort(key=lambda r: r['mtime'], reverse=True)
    # 内容判别只对最近的候选做，避免把上百个归档日志全读一遍
    kept = [r for r in runs[:max(limit * 2, 60)] if looks_like_training_log(r['path'])]
    return kept[:limit]


def resolve_log(explicit, runs_dir):
    """决定这次请求看哪份日志：显式参数 > current_run.txt > 最新的日志。"""
    if explicit and os.path.isfile(explicit):
        return explicit
    p = read_current_run(runs_dir)
    if p:
        return p
    runs = list_runs(runs_dir)
    return runs[0]['path'] if runs else (explicit or '')


def find_step_csv(out_dir, log_path):
    """给日志配出对应的 step_metrics.csv。

    日志名与 out 目录名的时间戳相差几秒（两边各自取当前时间），所以按
    “时间戳最接近”匹配，10 分钟内才算同一次运行。日志名里没有时间戳时
    才退回最新文件；有时间戳但对不上就返回 None —— 宁可让逐步损失用日志
    里的逐批损失重建，也不能把别的运行的曲线当成这一次的。
    """
    if not out_dir or not os.path.isdir(out_dir):
        return None
    cands = []
    for p in glob.glob(os.path.join(out_dir, '*', 'step_metrics.csv')):
        try:
            mtime = os.stat(p).st_mtime
        except OSError:
            continue
        run_ts = _parse_ts(os.path.basename(os.path.dirname(p)), '%Y-%m-%d_%H%M%S')
        cands.append((mtime, run_ts, p))
    if not cands:
        return None
    ts = log_timestamp(log_path)
    if ts is not None:
        close = [(abs(rt - ts), p) for _, rt, p in cands if rt is not None]
        close = [(d, p) for d, p in close if d <= 600]
        return min(close)[1] if close else None
    return max(cands)[2]


# ========================= 训练日志解析 =========================

RE_EPOCH = re.compile(
    r'^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{1,2}:[0-9]{2}:[0-9]{2}[.,]?[0-9]*\s+)?'
    r'Epoch\s+(?P<ep>\d+)/(?P<tot>\d+)'
    r'(?:\s*·\s*step\s+(?P<s0>\d+)~(?P<s1>\d+))?'
    r'\s*·\s*LR:\s*(?P<lr>[0-9.eE+-]+)\s*(?P<note>.*)$')
# 评估块（"【模型】训练损失: ..."）里的字段逐个抓，而不是一条大正则卡死顺序。
# 训练侧换个打印顺序、少打一项、把回归实验的 AUC 换成别的指标，这里都还能取到
# 剩下的字段；用一整条正则的话，任何一处对不上就整行丢弃，页面直接变空。
RE_EVAL_TAG = re.compile(r'\[模型\]')
_EVAL_FIELDS = (
    ('train_loss', re.compile(r'训练损失:\s*([0-9.]+)')),
    ('val_loss', re.compile(r'(?:验证集|测试集)损失:\s*([0-9.]+)')),
    ('auc_pooled', re.compile(r'AUC\(pooled\):\s*([0-9.]+)')),
    ('auc_daily', re.compile(r'AUC\(逐日\):\s*([0-9.n/a]+)')),
)
RE_DAILY = re.compile(
    r'逐日判别力:\s*IC\s*([+-]?[0-9.]+|n/a),\s*ICIR\s*([+-]?[0-9.]+|n/a),\s*'
    r'IC的t值\s*([+-]?[0-9.]+|n/a),\s*日Top(\d+)%命中率\s*([0-9.]+|n/a)\s*'
    r'\(统计天数\s*(\d+)\)')
RE_PROG = re.compile(r'训练进度:\s*([0-9.]+)%,\s*Loss:\s*([0-9.]+)')
RE_TOPRET = re.compile(r'Top(\d+)%收益:\s*([+-]?[0-9.]+)%')
RE_REAL = re.compile(r'平均实战收益率:\s*([+-]?[0-9.]+)%')
RE_NEWBEST = re.compile(r'新最佳模型（(.*?)）')
RE_BASELINE = re.compile(r'基线档案:\s*(.+)$')

# 评估块里凡是「名称: 数值」形式的打印，都当成指标自动收进 extras。
# 训练侧以后新增一项打印，页面上就会自动多出一条曲线，不需要改监控；
# 已经由上面白名单接管的字段不重复收，避免同一指标出现两次。
RE_GENERIC_KV = re.compile(
    r'^\s*([^\s:：][^:：]{0,23}?)\s*[:：]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
_GENERIC_SKIP = {
    '训练损失', '验证集损失', '测试集损失', 'AUC(pooled)', 'AUC(逐日)',
    'Top1%收益', '日Top1%收益', '实战收益率', '平均实战收益率',
    '训练进度', 'Loss', 'LR', '学习率',
}          # 带这些符号的行是字典 / 直方图 / 区间，拆出来的不是指标
_NOISE_LINE = '{}[]'
# 这些行是"破纪录/改善"的一次性播报，不是每轮都会有的周期指标，
# 收进来会让曲线出现孤立点（而且其中的 IC/AUC 已由白名单展示）。
_GENERIC_SKIP_LINE = ('新最佳模型', '改善', '每日统计', '已保存')


def _generic_metrics(ln):
    """把一行里的「名称: 数值」拆成 {指标名: 数值}。

    只认冒号紧跟纯数字的写法；阈值/区间类的键（含 <>）跳过，它们不是时间序列。
    """
    if any(s in ln for s in _GENERIC_SKIP_LINE):
        return {}
    if any(ch in ln for ch in _NOISE_LINE):
        # 行首的标签（如 "[模型]"）本身带方括号，先摘掉再判断有没有列表结构
        ln = re.sub(r'\[[^\]]{0,12}\]', ' ', ln)
        if any(ch in ln for ch in _NOISE_LINE):
            return {}
    found = {}
    for part in re.split(r'[,，]', ln):
        m = RE_GENERIC_KV.match(part)
        if not m:
            continue
        label = m.group(1).strip().lstrip('✓✅📈⚠').strip()
        # 按逗号拆开后，括号里的逗号会切出半截标签（如 "最大持仓4)】…"），
        # 这种带孤立右括号的片段一律不要。
        if not label or label in _GENERIC_SKIP or any(c in label for c in '<>)】）'):
            continue
        v = _num(m.group(2))
        if v is not None:
            found[label] = v
    return found

_CACHE = {'key': None, 'val': None}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_log(path, tail_lines=60, use_cache=True):
    if path and os.path.exists(path):
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if use_cache and _CACHE['key'] == key and _CACHE['val'] is not None:
            d = dict(_CACHE['val'])
            d['server_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            d['last_update_age'] = round(time.time() - st.st_mtime, 1)
            return d
    d = _parse_uncached(path, tail_lines)
    if path and os.path.exists(path):
        st = os.stat(path)
        _CACHE['key'] = (path, st.st_mtime_ns, st.st_size)
        _CACHE['val'] = dict(d)
    return d


def _parse_uncached(path, tail_lines=60):
    out = {'ok': False, 'log': path, 'log_size': 0, 'epochs': [], 'current': None,
           'total_epochs': None, 'baseline': None, 'tail': [], 'best': [],
           'server_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if not path or not os.path.exists(path):
        out['error'] = f'日志不存在: {path}'
        return out
    out['ok'] = True
    out['log_size'] = os.path.getsize(path)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    lines = re.split(r'[\r\n]+', raw)
    skip = ('InconsistentVersionWarning', 'warnings.warn', 'scikit-learn.org')
    out['tail'] = [l for l in lines if l.strip() and not any(s in l for s in skip)][-tail_lines:]

    cur = None
    for ln in lines:
        if not ln.strip() or any(s in ln for s in skip):
            continue
        m = RE_BASELINE.search(ln)
        if m:
            out['baseline'] = {'file': m.group(1).strip()}
        m = RE_EPOCH.match(ln.strip())
        if m:
            if cur is not None:
                out['epochs'].append(cur)
            g = m.groupdict()
            note = (g['note'] or '').strip()
            m_emb = re.search(r'\(embedding\s*([0-9.eE+-]+)\)', note)
            note = re.sub(r'\(embedding[^)]*\)', '', note).strip()
            cur = {'epoch': int(g['ep']), 'total': int(g['tot']), 'lr': _num(g['lr']),
                   'note': note, 'progress_pct': None}
            if m_emb:
                cur['lr_embed'] = _num(m_emb.group(1))
            if g['s0']:
                cur['step_from'] = int(g['s0'])
                cur['step_to'] = int(g['s1'])
            out['total_epochs'] = int(g['tot'])
            continue
        if cur is None:
            continue
        m = RE_PROG.search(ln)
        if m:
            pct, loss = _num(m.group(1)), _num(m.group(2))
            cur['progress_pct'] = pct
            cur['progress_loss'] = loss
            # 逐批损失也留一份：万一次运行的 step_metrics.csv 缺失（老日志、
            # 或换了写盘方式），逐步损失图还能用这些点重建，不至于只剩验证损失。
            if pct is not None and loss is not None:
                cur.setdefault('progress', []).append([pct, loss])
            continue
        if RE_EVAL_TAG.search(ln):
            for key, rx in _EVAL_FIELDS:
                fm = rx.search(ln)
                if not fm:
                    continue
                cur[key] = None if fm.group(1) == 'n/a' else _num(fm.group(1))
            # 同一行里除了上面这些白名单字段，可能还藏着别的指标（Prec@K 之类）
            ex = _generic_metrics(ln)
            if ex:
                cur.setdefault('extras', {}).update(ex)
            continue
        m = RE_DAILY.search(ln)
        if m:
            cur['ic'] = _num(m.group(1)) if m.group(1) != 'n/a' else None
            cur['icir'] = _num(m.group(2)) if m.group(2) != 'n/a' else None
            cur['ic_t'] = _num(m.group(3)) if m.group(3) != 'n/a' else None
            cur['topk_hit'] = _num(m.group(5)) if m.group(5) != 'n/a' else None
            cur['eval_days'] = int(m.group(6))
            continue
        m = RE_TOPRET.search(ln)
        if m:
            cur['top_return'] = _num(m.group(2))
            continue
        m = RE_REAL.search(ln)
        if m:
            cur['real_return'] = _num(m.group(1))
            continue
        m = RE_NEWBEST.search(ln)
        if m:
            out['best'].append({'epoch': cur['epoch'], 'what': m.group(1)})
            continue
        ex = _generic_metrics(ln)
        if ex:
            cur.setdefault('extras', {}).update(ex)

    if cur is not None:
        out['current'] = cur
        # 【不能再用 train_loss 非空判定"轮次已完成"】
        # 那一行只有训练循环里的评估块才会打。评估拆去 _eval_daemon.py 之后
        # （EVAL_EVERY=0），训练日志里根本没有这行 —— 于是每一轮都被过滤掉，
        # done_count 恒为 0。表现就是"总进度条永远 0%、进度条跑完也没变化"。
        # 训练损失改从"训练进度"行的实时 loss 兜底。
        if cur.get('train_loss') is None and cur.get('progress_loss') is not None:
            cur['train_loss'] = cur['progress_loss']
        out['epochs'].append(cur)
    out['epochs'] = [e for e in out['epochs'] if e.get('epoch') is not None]
    out['done_count'] = len(out['epochs'])
    out['log_step_series'] = steps_from_log(out['epochs'])
    out['last_update_age'] = round(time.time() - os.path.getmtime(path), 1)
    return out


def merge_eval_results(out, results_path):
    """把 _eval_daemon.py 写出的评估结果合并进 epochs。

    ## 为什么需要这一步

    评估已从训练循环里拆出去（训练每轮只存快照、不评估），所以训练日志里
    【不再有】训练损失/验证损失/AUC/IC 那些行。这些指标现在由独立的
    _eval_daemon.py 计算，逐条追加到 eval_results.jsonl。

    监控若只读训练日志，判别力那几条曲线就永远是空的 —— 这正是
    "页面好多信息不更新"的原因。

    按 epoch 号合并；同一轮若既有轻量档又有全量档，【全量档优先】。
    """
    if not results_path or not os.path.exists(results_path):
        return out

    by_ep = {}
    try:
        with open(results_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get('error'):
                    continue
                ep = rec.get('epoch')
                if ep is None:
                    continue
                prev = by_ep.get(ep)
                if prev is None or (rec.get('mode') == 'full' and prev.get('mode') != 'full'):
                    by_ep[ep] = rec
    except Exception:
        return out

    out['eval_count'] = len(by_ep)
    out['eval_results_path'] = results_path

    for e in out.get('epochs', []):
        r = by_ep.get(e.get('epoch'))
        if not r:
            continue
        # 【只在非空时覆盖】。eval 记录里某些字段可能为 None（例如 daemon
        # 从 checkpoint 读 train_loss 拿不到时），无条件覆盖会把训练日志里
        # 本来已经解析出来的值抹掉，页面上就显示成 "—"。
        for dst, src in (('train_loss', 'train_loss'),
                         ('auc_pooled', 'auc'),
                         ('auc_daily', 'daily_auc'),
                         ('ic', 'ic_mean'),
                         ('icir', 'icir'),
                         ('ic_t', 'ic_t'),
                         ('top_return', 'top_return'),
                         ('val_loss', 'test_loss')):
            v = r.get(src)
            if v is not None:
                e[dst] = v
        e['eval_mode'] = r.get('mode')
        e['eval_n'] = r.get('n_eval')

    # daemon 可能跑在训练日志前面：已评估但日志里还没有的轮次补成独立条目，
    # 否则最新几个点画不出来
    known = {e.get('epoch') for e in out.get('epochs', [])}
    for ep in sorted(by_ep):
        if ep in known:
            continue
        r = by_ep[ep]
        out.setdefault('epochs', []).append({
            'epoch': ep, 'total': out.get('total_epochs') or 0, 'lr': r.get('lr'),
            'note': '仅评估', 'progress_pct': None,
            'train_loss': r.get('train_loss'),
            'val_loss': r.get('test_loss'),
            'auc_pooled': r.get('auc'), 'auc_daily': r.get('daily_auc'),
            'ic': r.get('ic_mean'), 'icir': r.get('icir'), 'ic_t': r.get('ic_t'),
            'top_return': r.get('top_return'),
            'eval_mode': r.get('mode'), 'eval_n': r.get('n_eval'),
        })
    out['epochs'].sort(key=lambda e: e.get('epoch') or 0)
    # 合并完再重算已完成轮数 —— parse_log 里算的时候还不知道 daemon 的结果，
    # 而"完成"的真正含义是"这一轮的快照已经跑完并（可能）被评估"。
    out['done_count'] = max((e.get('epoch') or 0) for e in out['epochs']) if out['epochs'] else 0
    return out


def load_compare_series(specs):
    """加载【对照实验】的评估记录，供前端叠加曲线。

    消融实验需要把两条曲线画在一起看是否重合：
        冻结 embedding（正在跑） vs 解冻 embedding（已跑完）
    其余配置完全相同，所以唯一差异就是 embedding 是否参与训练。

    specs: [(label, path), ...]
    返回: [{'label': ..., 'points': [{epoch, train_loss, val_loss, ...}]}, ...]

    字段名要与前端 METRICS 里的 key 对齐，否则曲线画不出来。
    """
    out = []
    for label, path in specs:
        if not path or not os.path.exists(path):
            print(f"  [compare] 跳过（不存在）: {label} -> {path}")
            continue
        pts = {}
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get('error'):
                        continue
                    ep = r.get('epoch')
                    if ep is None:
                        continue
                    prev = pts.get(ep)
                    # 同一轮若轻量/全量都有，全量优先
                    if prev is not None and not (r.get('mode') == 'full' and prev.get('_mode') != 'full'):
                        continue
                    pts[ep] = {
                        '_mode': r.get('mode'),
                        'epoch': ep,
                        'lr': r.get('lr'),
                        'train_loss': r.get('train_loss'),
                        'val_loss': r.get('test_loss'),
                        'auc_pooled': r.get('auc'),
                        'auc_daily': r.get('daily_auc'),
                        'ic': r.get('ic_mean'),
                        'icir': r.get('icir'),
                        'ic_t': r.get('ic_t'),
                        'top_return': r.get('top_return'),
                    }
        except Exception as e:
            print(f"  [compare] {label} 读取失败: {e}")
            continue
        rows = sorted(pts.values(), key=lambda x: x['epoch'])
        for r in rows:
            r.pop('_mode', None)

        # ---- 修正 LR 的【时点偏移】----
        # checkpoint 里存的是 optimizer 在第 N 轮【结束后】的 lr，也就是第 N+1 轮
        # 实际使用的值；而主曲线（实线）的 lr 来自训练日志 "Epoch N, LR: x"，
        # 是第 N 轮【开始时】的值。两者差一轮，直接画会呈现"明明调度相同却错开"的
        # 假象（本次消融实验的 LR 校验就撞上了）。
        # 这里把 checkpoint 的 lr 往后挪一轮，使两条线的横坐标都表示"该轮使用的 LR"。
        for i in range(len(rows) - 1, 0, -1):
            rows[i]['lr'] = rows[i - 1].get('lr')
        if rows:
            rows[0]['lr'] = None      # 第 1 轮无前一个 checkpoint 可推
        out.append({'label': label, 'n': len(rows), 'path': path, 'points': rows})
        print(f"  [compare] {label}: {len(rows)} 个点  ({os.path.basename(os.path.dirname(path))})")
    return out


def steps_from_log(epochs, max_points=1500):
    """用日志里的『训练进度: x%, Loss: y』行重建逐步损失序列。

    正常情况下逐步损失来自 train.py 写的 step_metrics.csv；老日志、或者哪天
    换了写盘方式没有这个 CSV 时，用这里的重建结果顶上，图不会变成空白。
    横坐标按该轮的 step 区间把百分比换算成全局步数。
    """
    pts = []
    for e in epochs or []:
        s0, s1 = e.get('step_from'), e.get('step_to')
        prog = e.get('progress') or []
        if not s0 or not s1 or s1 < s0 or not prog:
            continue
        span = s1 - s0 + 1
        for pct, loss in prog:
            if loss is None:
                continue
            pts.append([s0 + int(round(pct / 100.0 * span)) - 1, loss])
    if len(pts) > max_points:
        stride = len(pts) // max_points + 1
        pts = pts[::stride]
    return pts


# ========================= 页面 =========================

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EquiNet 训练监控</title>
<script src="static/echarts.min.js"></script>
<style>
 :root{--bg:#0d1017;--panel:#151922;--panel2:#1b2130;--line:#262d3a;
       --fg:#e8ecf4;--dim:#8b93a7;--acc:#4ea1ff;--ok:#3ddc97;--warn:#ffb454;--bad:#ff6b6b;}
 *{box-sizing:border-box}
 html,body{margin:0;padding:0}
 body{background:var(--bg);color:var(--fg);
      font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
 header{padding:9px 18px;border-bottom:1px solid var(--line);
        display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;
        position:sticky;top:0;background:var(--bg);z-index:20}
 h1{font-size:14px;margin:0;font-weight:600}
 .meta{color:var(--dim);font-size:12px}
 .sel{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
      border-radius:5px;font:inherit;font-size:12px;padding:2px 6px;max-width:340px}
 .wrap{padding:12px 18px 40px;display:grid;gap:12px;
       grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
 .card h2{font-size:11px;margin:0 0 8px;color:var(--dim);font-weight:600;
          letter-spacing:.08em;text-transform:uppercase}
 .full{grid-column:1/-1}
 table{border-collapse:collapse;width:100%}
 td{padding:2px 0;vertical-align:top}
 td.k{color:var(--dim);width:56%}
 td.v{font-variant-numeric:tabular-nums}
 pre{margin:0;white-space:pre-wrap;word-break:break-all;color:#b9c0d0;
     font-size:11.5px;max-height:240px;overflow:auto}
 .live{color:var(--ok)} .stale{color:var(--warn)} .dead{color:var(--bad)}
 .bar{height:6px;background:#222733;border-radius:3px;overflow:hidden;margin:5px 0}
 .bar>i{display:block;height:100%;background:linear-gradient(90deg,#2d6cdf,#4ea1ff)}
 .bar.ep>i{background:linear-gradient(90deg,#1f7a5a,#3ddc97)}
 .btns{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
 .btn{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
      padding:3px 11px;border-radius:5px;cursor:pointer;font:inherit;font-size:12px}
 .btn:hover{border-color:#3b465c;color:var(--fg)}
 .btn.on{background:#1d3a5c;border-color:var(--acc);color:#cfe3ff}
 #bigchart{width:100%;height:440px}
 /* 资源图与训练曲线同高 —— 230px 时 Y 轴太挤，四条线的细节看不出来 */
 #reschart{width:100%;height:440px}
 #stepchart{width:100%;height:380px}
 body.fs .wrap{display:none}
 body.fs #fsWrap{display:block}
 #fsWrap{display:none;position:fixed;inset:0;background:var(--bg);z-index:100;padding:8px}
 #fsWrap .fsbar{display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap}
 #fschart{width:100%;height:calc(100vh - 60px)}
 .hint{color:var(--dim);font-size:11px;margin-top:6px}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:0 14px}
 .rs{display:flex;align-items:baseline;gap:8px;margin:3px 0}
 .rs b{font-variant-numeric:tabular-nums;font-weight:600}
 .rs span{color:var(--dim);font-size:11px}
</style>
</head>
<body>
<header>
  <h1>EquiNet 训练监控</h1>
  <span class="meta" id="conn">连接中…</span>
  <span class="meta" id="clock"></span>
  <select id="runsel" class="sel" title="选择要看的运行；自动 = 跟随最新/current_run.txt"></select>
  <span class="meta" id="logpath"></span>
</header>
<div id="jserr" style="display:none;margin:8px 18px;padding:8px 12px;border:1px solid #ff6b6b;
     border-radius:6px;background:#2a1518;color:#ffb3b3;font-size:12px;white-space:pre-wrap"></div>

<div class="wrap">
  <div class="card full">
    <h2>总进度</h2>
    <div id="epochline" class="v" style="font-size:15px">—</div>
    <div class="bar ep"><i id="epbar" style="width:0%"></i></div>
    <h2 style="margin-top:10px">当前轮次内进度<span id="stepnote" style="font-weight:400;color:#8b93a7;font-size:11px"></span></h2>
    <div id="stepline" class="v">—</div>
    <div class="bar"><i id="stepbar" style="width:0%"></i></div>
  </div>

  <div class="card full">
    <h2>系统资源（最近 5 分钟 · 每 2.5 秒采样）</h2>
    <div class="grid2">
      <div>
        <div class="rs"><span>CPU</span><b id="r_cpu">—</b><span id="r_load">—</span></div>
        <div class="rs"><span>内存</span><b id="r_mem">—</b><span id="r_memgb">—</span></div>
        <div class="rs"><span>磁盘</span><b id="r_disk">—</b><span id="r_diskgb">—</span></div>
      </div>
      <div>
        <div class="rs"><span>GPU</span><b id="r_gpu">—</b><span id="r_gputmp">—</span></div>
        <div class="rs"><span>显存</span><b id="r_vram">—</b><span id="r_vramgb">—</span></div>
        <div class="rs"><span>GPU进程</span><b id="r_gproc">—</b></div>
      </div>
    </div>
    <div id="reschart"></div>
    <div class="hint">CPU% / GPU利用率% / 内存% / 显存% —— 与主图的「资源」指标组同源。</div>
  </div>

  <div class="card full">
    <h2>逐步损失（训练每步 · 验证每周期）</h2>
    <div id="stepchart"></div>
    <div class="hint">训练损失每一步一个点；验证损失要走全量验证集，只在每个测评周期末算一次。
      <span id="stepsrc"></span></div>
  </div>

  <div class="card full">
    <h2>训练曲线（滚轮缩放 · 拖动平移 · 双击复位 · 双击卡片进全屏）</h2>
    <div class="btns" id="metricBtns"></div>
    <div id="bigchart"></div>
    <div class="hint">滚轮缩放；按住拖动平移；底部滑块缩放；图例可点击隐藏曲线。
      资源图默认显示最近 5 分钟，同样可缩放。</div>
  </div>

  <div class="card full">
    <h2>其它指标（从日志自动识别）</h2>
    <div class="btns" id="extraBtns"></div>
    <div id="extrachart" style="width:100%;height:300px"></div>
    <table id="extratbl" style="margin-top:8px"></table>
    <div class="hint">评估块里任何「名称: 数值」的打印都会自动收进来（最多 12 个），
      训练侧新增指标不需要改监控；按钮切换看哪一个，图例仍可点击隐藏。</div>
  </div>

  <div class="card">
    <h2>当前指标</h2>
    <table>
      <tr><td class="k">训练损失</td><td class="v" id="tloss">—</td>
          <td class="k">验证损失</td><td class="v" id="vloss">—</td></tr>
      <tr><td class="k">AUC(pooled)</td><td class="v" id="aucp">—</td>
          <td class="k">AUC(逐日)</td><td class="v" id="aucd">—</td></tr>
      <tr><td class="k">IC</td><td class="v" id="ic">—</td>
          <td class="k">ICIR</td><td class="v" id="icir">—</td></tr>
      <tr><td class="k">IC 的 t 值</td><td class="v" id="ict">—</td>
          <td class="k">日Top-K 命中率</td><td class="v" id="hit">—</td></tr>
      <tr><td class="k">Top-K 收益</td><td class="v" id="topret">—</td>
          <td class="k">实战收益率</td><td class="v" id="realret">—</td></tr>
      <tr><td class="k">当前 LR</td><td class="v" id="lr">—</td>
          <td class="k">阶段</td><td class="v" id="note">—</td></tr>
      <tr><td class="k">统计天数</td><td class="v" id="days">—</td>
          <td class="k">已评估轮数</td><td class="v" id="nep">—</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>基线 / 最佳记录</h2>
    <div id="baseline" class="v">—</div>
    <div id="best" class="v" style="max-height:190px;overflow:auto;font-size:11.5px;margin-top:8px">—</div>
  </div>

  <div class="card full">
    <h2>日志尾部</h2>
    <pre id="tail">—</pre>
  </div>
</div>

<div id="fsWrap">
  <div class="fsbar">
    <div class="btns" id="metricBtnsFs" style="margin:0"></div>
    <button class="btn" onclick="toggleFs(false)">退出全屏 (Esc)</button>
  </div>
  <div id="fschart"></div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = (v,n=4) => (v===null||v===undefined||Number.isNaN(v))?'—':Number(v).toFixed(n);
const sgn = (v,n=4) => (v===null||v===undefined||Number.isNaN(v))?'—':(v>=0?'+':'')+Number(v).toFixed(n);

const METRICS = {
  loss: {name:'损失', series:[
      {key:'train_loss', name:'训练损失', color:'#4ea1ff'},
      {key:'val_loss',   name:'验证损失', color:'#ffb454'}], yname:'Loss'},
  auc:  {name:'AUC', series:[
      {key:'auc_pooled', name:'AUC(pooled)', color:'#ffb454'},
      {key:'auc_daily',  name:'AUC(逐日)',   color:'#4ea1ff'}], yname:'AUC'},
  ic:   {name:'判别力', series:[
      {key:'ic',   name:'IC',   color:'#4ea1ff'},
      {key:'icir', name:'ICIR', color:'#3ddc97'},
      {key:'ic_t', name:'IC的t值', color:'#c792ea'}], yname:'值'},
  ret:  {name:'收益', series:[
      {key:'top_return', name:'Top1%收益(%)', color:'#3ddc97'},
      {key:'real_return',name:'实战收益率(%)', color:'#ffb454'}], yname:'%'},
  lr:   {name:'学习率', series:[
      {key:'lr', name:'LR', color:'#3ddc97'}], yname:'LR', log:true},
  // 注意：这里【没有】「资源」组。
  // 资源图只在上面「系统资源」卡片里出现一次 —— 之前这里也放了一份，
  // 点按钮后屏幕上会同时出现两张一样的图，属于重复展示。
};
let curMetric = 'loss';
const charts = {};
const views = {};
const STATE = {epochs:[], res:[], compare:[], run:null, runs:[],
               extraRows:[], extraPick:null};
// 资源图用的固定配置（它不在指标切换器里，是独立的一张图）
const RES_METRIC = {name:'资源', res:true, series:[
    {key:'cpu_pct',  name:'CPU%',      color:'#4ea1ff'},
    {key:'gpu_util', name:'GPU利用率%', color:'#3ddc97'},
    {key:'mem_pct',  name:'内存%',      color:'#ffb454'},
    {key:'vram_pct', name:'显存%',      color:'#c792ea'}], yname:'%'};

function seriesData(metric){
  const M = (metric === 'res') ? RES_METRIC : METRICS[metric];
  if(M.res) return STATE.res;
  return STATE.epochs;
}

function makeOption(metric, compact){
  const M = (metric === 'res') ? RES_METRIC : METRICS[metric];
  const data = seriesData(metric);
  const isRes = !!M.res;
  const isLog = !!M.log && !isRes;
  const series = M.series.map(s => ({
      name:s.name, type:'line', showSymbol:false, symbol:'circle', symbolSize:5,
      sampling:'lttb', lineStyle:{width:1.6, color:s.color}, itemStyle:{color:s.color},
      data: data.map(e => {
          const v = e[s.key];
          // 对数轴不能有 0 或负值。训练日志用 %.6f 打印 LR，预热的起始值
          // (3e-7) 会被打成 "0.000000"，解析回来就是 0 —— 若不剔除，
          // ECharts 对含 0 的对数轴会整条序列都画不出来（本次的 bug）。
          // 只丢这一个点，其余轮次不受影响。
          if (isLog && (v === undefined || v === null || !(v > 0))) return null;
          return [isRes ? e.t*1000 : e.epoch, (v === undefined ? null : v)];
      }).filter(p => p !== null),
  }));

  // ---- 消融对照：把其它 run 的同名指标画成虚线叠上去 ----
  // 实线 = 当前正在跑的（冻结）；虚线 = 对照（解冻）。同色不同线型，
  // 这样"是否重合"一眼能看出来，而不用在图例里找颜色。
  if(!isRes && STATE.compare && STATE.compare.length){
    for(const cmp of STATE.compare){
      const pts = cmp.points || [];
      if(!pts.length) continue;
      for(const s of M.series){
        series.push({
          name: s.name + '·' + cmp.label,
          type:'line', showSymbol:false, symbol:'circle', symbolSize:4,
          sampling:'lttb',
          lineStyle:{width:1.2, type:'dashed', color:s.color, opacity:0.75},
          itemStyle:{color:s.color, opacity:0.75},
          data: pts.map(e => {
            const v = e[s.key];
            if(isLog && (v === undefined || v === null || !(v > 0))) return null;
            return [e.epoch, (v === undefined ? null : v)];
          }).filter(p => p !== null),
        });
      }
    }
  }
  const xAxis = isRes
    ? {type:'time', name:'时间', nameLocation:'middle', nameGap:22,
       nameTextStyle:{color:'#8b93a7', fontSize:11},
       axisLine:{lineStyle:{color:'#39415a'}},
       axisLabel:{color:'#8b93a7', fontSize:10,
                  formatter:{value: v => {
                     const d=new Date(v);
                     return ('0'+d.getHours()).slice(-2)+':'+
                            ('0'+d.getMinutes()).slice(-2)+':'+
                            ('0'+d.getSeconds()).slice(-2);}}},
       splitLine:{lineStyle:{color:'#1e2432'}}}
    : {type:'value', name:'Epoch', nameLocation:'middle', nameGap:26,
       nameTextStyle:{color:'#8b93a7', fontSize:12},
       min:'dataMin', max:'dataMax', scale:true,
       axisLine:{lineStyle:{color:'#39415a'}},
       axisLabel:{color:'#8b93a7', fontSize:11},
       splitLine:{lineStyle:{color:'#1e2432'}}};

  const yAxis = {type: (M.log && !isRes) ? 'log' : 'value', name:M.yname,
    nameLocation:'middle', nameGap:compact?38:48,
    nameTextStyle:{color:'#8b93a7', fontSize:12},
    scale: !isRes, min: isRes?0:undefined, max: isRes?100:undefined,
    axisLine:{lineStyle:{color:'#39415a'}},
    axisLabel:{color:'#8b93a7', fontSize:11,
      formatter: v => (Math.abs(v)<1e-3 && v!==0) ? v.toExponential(1) : v},
    splitLine:{lineStyle:{color:'#1e2432'}}};

    // 数据窗口默认显示最近 RES_WINDOW_SEC 秒，但同样支持缩放/平移：
    // 训练是间歇性的，某些时刻 GPU/CPU 会掉到 0，放大看细节很有必要。
    const dz = [
      {type:'inside', xAxisIndex:0, filterMode:'none', zoomOnMouseWheel:true,
       moveOnMouseMove:true, moveOnMouseWheel:false},
      {type:'slider', xAxisIndex:0, height:compact?14:22, bottom:compact?4:16,
       borderColor:'#30384a', backgroundColor:'#151922',
       fillerColor:'rgba(78,161,255,.15)', handleStyle:{color:'#4ea1ff'},
       textStyle:{color:'#8b93a7', fontSize:10}}];

  return {
    animation:false, backgroundColor:'transparent',
    grid:{left:compact?56:70, right:compact?16:36, top:compact?34:44,
          bottom: isRes ? 34 : (compact?44:74)},
    tooltip:{trigger:'axis', axisPointer:{type:'cross', lineStyle:{color:'#556080'}},
      backgroundColor:'rgba(20,24,33,.96)', borderColor:'#30384a',
      textStyle:{color:'#e8ecf4', fontSize:12},
      formatter: ps => {
        let head;
        if(isRes){
          const d=new Date(ps[0].axisValue);
          head = ('0'+d.getHours()).slice(-2)+':'+('0'+d.getMinutes()).slice(-2)+':'+('0'+d.getSeconds()).slice(-2);
        } else { head = 'Epoch '+ps[0].axisValue; }
        let s='<b>'+head+'</b>';
        ps.forEach(p=>{ if(p.value&&p.value[1]!==null&&p.value[1]!==undefined)
          s+='<br>'+p.marker+p.seriesName+': <b>'+
             (Math.abs(p.value[1])<1e-3&&p.value[1]!==0?p.value[1].toExponential(3)
              :Number(p.value[1]).toFixed(isRes?1:5))+'</b>'; });
        return s;
      }},
    legend:{top:4, left:compact?56:70, textStyle:{color:'#b9c0d0', fontSize:12},
            inactiveColor:'#4a5165', itemWidth:16, itemHeight:8},
    toolbox:{right:compact?6:12, top:2, feature:{
        saveAsImage:{title:'保存图片', backgroundColor:'#0d1017'},
        restore:{title:'重置'}}, iconStyle:{borderColor:'#8b93a7'}},
    xAxis:xAxis, yAxis:yAxis, dataZoom:dz, series:series,
  };
}

function makeStepOption(){
  const train = STATE.steps || [];
  const val = (STATE.epochs || [])
      .filter(e => e.step_to != null && e.val_loss != null)
      .map(e => [e.step_to, e.val_loss]);
  return {
    animation:false, backgroundColor:'transparent',
    grid:{left:70,right:36,top:38,bottom:56},
    tooltip:{trigger:'axis', axisPointer:{type:'cross', lineStyle:{color:'#556080'}},
      backgroundColor:'rgba(20,24,33,.96)', borderColor:'#30384a',
      textStyle:{color:'#e8ecf4', fontSize:12},
      formatter: ps => {
        let s = 'Step ' + ps[0].axisValue;
        ps.forEach(p => { if (p.value && p.value[1] != null)
          s += '<br>' + p.marker + p.seriesName + ': <b>' + Number(p.value[1]).toFixed(4) + '</b>'; });
        return s;
      }},
    legend:{top:4, left:70, textStyle:{color:'#b9c0d0', fontSize:12}, itemWidth:16, itemHeight:8},
    xAxis:{type:'value', name:'step', nameLocation:'middle', nameGap:26,
      nameTextStyle:{color:'#8b93a7', fontSize:12}, scale:true,
      axisLine:{lineStyle:{color:'#39415a'}},
      axisLabel:{color:'#8b93a7', fontSize:11},
      splitLine:{lineStyle:{color:'#1e2432'}}},
    yAxis:{type:'value', name:'Loss', nameLocation:'middle', nameGap:50,
      nameTextStyle:{color:'#8b93a7', fontSize:12}, scale:true,
      axisLine:{lineStyle:{color:'#39415a'}},
      axisLabel:{color:'#8b93a7', fontSize:11},
      splitLine:{lineStyle:{color:'#1e2432'}}},
    dataZoom:[{type:'inside', xAxisIndex:0, filterMode:'none'},
              {type:'slider', xAxisIndex:0, height:20, bottom:12,
               borderColor:'#30384a', backgroundColor:'#151922',
               fillerColor:'rgba(78,161,255,.15)', handleStyle:{color:'#4ea1ff'},
               textStyle:{color:'#8b93a7', fontSize:10}}],
    series:[
      {name:'训练损失（每步）', type:'line', showSymbol:false, sampling:'lttb',
       lineStyle:{width:1.2, color:'#4ea1ff'}, itemStyle:{color:'#4ea1ff'}, data:train},
      {name:'验证损失（周期末）', type:'line', showSymbol:true, symbolSize:6,
       lineStyle:{width:2, color:'#ffb454'}, itemStyle:{color:'#ffb454'}, data:val},
    ],
  };
}

function updateStepChart(){
  const c = charts['stepchart']; if(!c) return;
  c.setOption(makeStepOption(), {replaceMerge:['series','dataZoom']});
}

function initChart(elId, metric, compact){
  const el = $(elId); if(!el) return null;
  const c = echarts.init(el, null, {renderer:'canvas'});
  c.__metric = metric; c.__compact = compact; c.__elId = elId;
  c.getZr().on('dblclick', ()=>c.dispatchAction({type:'dataZoom', start:0, end:100}));
  c.on('dataZoom', ()=>{ const o=c.getOption().dataZoom;
    if(o&&o.length) views[elId]=o.map(d=>({start:d.start,end:d.end})); });
  charts[elId]=c; return c;
}

function updateChart(elId){
  const c=charts[elId]; if(!c) return;
  c.setOption(makeOption(c.__metric, c.__compact), {replaceMerge:['series','dataZoom']});
  const v=views[elId];
  if(v&&v.length){
    c.dispatchAction({type:'dataZoom', dataZoomIndex:0, start:v[0].start, end:v[0].end});
    if(v[1]) c.dispatchAction({type:'dataZoom', dataZoomIndex:1, start:v[1].start, end:v[1].end});
  }
}

function buildButtons(containerId, onPick){
  const box=$(containerId); box.innerHTML='';
  Object.keys(METRICS).forEach(k=>{
    const b=document.createElement('button');
    b.className='btn'+(k===curMetric?' on':''); b.textContent=METRICS[k].name;
    b.onclick=()=>{ curMetric=k; onPick(k); };
    box.appendChild(b);
  });
}
function refreshButtons(){
  ['metricBtns','metricBtnsFs'].forEach(id=>{
    const box=$(id); if(!box) return;
    [...box.children].forEach((b,i)=>{
      b.className='btn'+(Object.keys(METRICS)[i]===curMetric?' on':''); });
  });
}
function toggleFs(on){
  document.body.classList.toggle('fs', on);
  if(on){
    $('fsWrap').style.display='block';
    initChart('fschart', curMetric, false);
    setTimeout(()=>{ charts['fschart']&&charts['fschart'].resize(); updateChart('fschart'); },30);
  } else $('fsWrap').style.display='none';
}
document.addEventListener('keydown', e=>{ if(e.key==='Escape'&&document.body.classList.contains('fs')) toggleFs(false); });

function renderRes(s){
  if(!s) return;
  // 统一取值助手：null/undefined/NaN 一律显示 —，绝不调用 .toFixed 导致整页崩掉
  const ok = v => (v===null||v===undefined||Number.isNaN(v)) ? null : Number(v);
  const set=(id,v)=>{ const e=$(id); if(e) e.textContent=(v===null||v===undefined)?'—':v; };
  const pct=(v,d=1)=>{ const x=ok(v); return x===null?'—':x.toFixed(d)+'%'; };

  set('r_cpu', pct(s.cpu_pct));
  const L=ok(s.load1), L5=ok(s.load5), L15=ok(s.load15);
  set('r_load', L===null?'':`负载 ${L} / ${L5} / ${L15} · ${s.cpu_count||'?'} 核`);

  const m=s.mem||{};
  set('r_mem', pct(s.mem_pct));
  set('r_memgb', ok(m.total_gb)===null?'':`${ok(m.used_gb)} / ${ok(m.total_gb)} GB（可用 ${ok(m.avail_gb)} GB）`);

  const d=s.disk||{};
  set('r_disk', pct(d.pct));
  set('r_diskgb', ok(d.total_g)===null?'':`${ok(d.used_g)} / ${ok(d.total_g)} GB（剩 ${ok(d.free_g)} GB）`);

  const g=s.gpu||{};
  set('r_gpu', ok(g.util)===null?'—':ok(g.util).toFixed(0)+'%');
  const tc=ok(g.temp_c), pw=ok(g.power_w), pl=ok(g.power_limit_w), fan=ok(g.fan_pct);
  set('r_gputmp', tc===null?'':`${tc.toFixed(0)}°C · ${pw===null?'?':pw.toFixed(0)}W / ${pl===null?'?':pl.toFixed(0)}W · 风扇 ${fan===null?'?':fan.toFixed(0)}%`);
  set('r_vram', pct(g.mem_pct));
  set('r_vramgb', ok(g.mem_total_gb)===null?'':`${ok(g.mem_used_gb)} / ${ok(g.mem_total_gb)} GB`);
  set('r_gproc', (g.procs&&g.procs.length)
      ? g.procs.map(p=>`PID ${p.pid}: ${(p.mem_mb/1024).toFixed(2)} GB`).join(' · ') : '无');
}

function fmtClock(t){
  if(!t) return '';
  const d=new Date(t*1000);
  return ('0'+(d.getMonth()+1)).slice(-2)+'-'+('0'+d.getDate()).slice(-2)+' '+
         ('0'+d.getHours()).slice(-2)+':'+('0'+d.getMinutes()).slice(-2);
}

// 运行切换器：列表来自服务端扫 train_logs/ 的结果，选中项通过 ?run= 传给接口，
// 因此回看历史实验不需要重启服务，也不需要改代码。
function renderRuns(d){
  const sel=$('runsel'); if(!sel) return;
  const runs=d.runs||[];
  const sig=runs.map(r=>r.name).join('|');
  if(sel.__sig!==sig){
    sel.__sig=sig;
    sel.innerHTML='<option value="__auto__">自动跟随（最新 / current_run.txt）</option>'+
      runs.map(r=>`<option value="${r.name}">${r.name.replace(/^train_/,'').replace(/\.log$/,'')} · ${fmtClock(r.mtime)}</option>`).join('');
  }
  sel.value = STATE.run || '__auto__';
  if(sel.__bound) return;
  sel.__bound=true;
  sel.onchange=()=>{ STATE.run = sel.value==='__auto__' ? null : sel.value; tick(); };
}

const EXTRA_MAX = 12;

// 自动识别的指标：后端把评估块里「名称: 数值」都收进 epoch.extras，
// 这里不预设任何键名，出现什么就画什么 —— 训练侧加指标不用改页面。
function extraKeysOf(epochs){
  const seen=[];
  for(const e of (epochs||[])){
    const x=e.extras||{};
    for(const k in x){ if(seen.indexOf(k)<0) seen.push(k); }
  }
  return seen.slice(0, EXTRA_MAX);
}

function extraRowsOf(epochs){
  return extraKeysOf(epochs).map(k=>{
    const pts=[];
    for(const e of epochs){
      const x=e.extras||{};
      if(x[k]!==undefined && x[k]!==null) pts.push([e.epoch, x[k]]);
    }
    return {k, pts};
  });
}

function buildExtraButtons(){
  const box=$('extraBtns'); if(!box) return;
  const sig=STATE.extraRows.map(r=>r.k).join('|');
  if(box.__sig!==sig){
    box.__sig=sig; box.innerHTML='';
    STATE.extraRows.forEach(r=>{
      const b=document.createElement('button'); b.className='btn'; b.textContent=r.k;
      b.onclick=()=>{ STATE.extraPick=r.k; buildExtraButtons(); updateExtraChart(); };
      box.appendChild(b);
    });
  }
  [...box.children].forEach(b=>{
    b.className='btn'+(b.textContent===STATE.extraPick?' on':''); });
}

function updateExtraChart(){
  const el=$('extrachart'); if(!el) return;
  const row=STATE.extraRows.find(r=>r.k===STATE.extraPick);
  if(!row){ return; }
  if(!charts['extrachart']) charts['extrachart']=echarts.init(el,null,{renderer:'canvas'});
  const c=charts['extrachart'];
  c.setOption({
    animation:false, backgroundColor:'transparent',
    grid:{left:70,right:36,top:20,bottom:56},
    tooltip:{trigger:'axis', backgroundColor:'rgba(20,24,33,.96)', borderColor:'#30384a',
      textStyle:{color:'#e8ecf4', fontSize:12},
      formatter: ps=>{ let s='<b>Epoch '+ps[0].axisValue+'</b>';
        ps.forEach(p=>{ if(p.value&&p.value[1]!==null)
          s+='<br>'+p.marker+p.seriesName+': <b>'+Number(p.value[1]).toFixed(5)+'</b>'; });
        return s; }},
    xAxis:{type:'value', name:'Epoch', nameLocation:'middle', nameGap:26, scale:true,
      nameTextStyle:{color:'#8b93a7', fontSize:12},
      axisLine:{lineStyle:{color:'#39415a'}}, axisLabel:{color:'#8b93a7', fontSize:11},
      splitLine:{lineStyle:{color:'#1e2432'}}},
    yAxis:{type:'value', name:row.k, nameLocation:'middle', nameGap:52, scale:true,
      nameTextStyle:{color:'#8b93a7', fontSize:11},
      axisLine:{lineStyle:{color:'#39415a'}}, axisLabel:{color:'#8b93a7', fontSize:11},
      splitLine:{lineStyle:{color:'#1e2432'}}},
    dataZoom:[{type:'inside', xAxisIndex:0, filterMode:'none'},
              {type:'slider', xAxisIndex:0, height:20, bottom:12,
               borderColor:'#30384a', backgroundColor:'#151922',
               fillerColor:'rgba(78,161,255,.15)', handleStyle:{color:'#4ea1ff'},
               textStyle:{color:'#8b93a7', fontSize:10}}],
    series:[{name:row.k, type:'line', showSymbol:true, symbolSize:5, connectNulls:true,
      lineStyle:{width:1.8, color:'#3ddc97'}, itemStyle:{color:'#3ddc97'}, data:row.pts}],
  }, {replaceMerge:['series','dataZoom']});
}

function renderExtraTable(){
  const tb=$('extratbl'); if(!tb) return;
  if(!STATE.extraRows.length){
    tb.innerHTML='<tr><td class="k">（日志里还没有可识别的「名称: 数值」指标）</td></tr>';
    return;
  }
  tb.innerHTML=STATE.extraRows.map(r=>{
    const n=r.pts.length, last=n?r.pts[n-1]:null, prev=n>1?r.pts[n-2]:null;
    const delta=(prev&&last)?(last[1]-prev[1]):null;
    return `<tr><td class="k">${r.k}</td><td class="v">${last?fmt(last[1],4):'—'}</td>`+
      `<td class="k" style="text-align:right">${last?('Epoch '+last[0]):''}`+
      (delta===null?'':` · Δ ${delta>=0?'+':''}${delta.toFixed(4)}`)+'</td></tr>';
  }).join('');
}

function renderExtras(d){
  STATE.extraRows = extraRowsOf(d.epochs||[]);
  if(!STATE.extraPick || !STATE.extraRows.some(r=>r.k===STATE.extraPick))
    STATE.extraPick = STATE.extraRows.length ? STATE.extraRows[0].k : null;
  buildExtraButtons();
  updateExtraChart();
  renderExtraTable();
}

function render(d){
  // 先把连接状态更新掉 —— 之前它排在 renderRes/图表代码后面，
  // 一旦那些地方抛异常，页面就会永远停在「连接中」，掩盖真正的错误。
  const conn=$('conn');
  try{
    const age=d.last_update_age;
    conn.textContent=(age<120?'● 实时':(age<900?'◐ 稍旧':'○ 停滞'))+`（日志 ${age}s 前更新）`;
    conn.className='meta '+(age<120?'live':(age<900?'stale':'dead'));
  }catch(e){ conn.textContent='● 已连接'; conn.className='meta live'; }

  $('logpath').textContent = (d.log ? d.log.split('/').pop() : '') +
    ' · 逐步指标: ' + (d.steps_csv ? d.steps_csv.split('/').slice(-2).join('/') : '未找到');
  renderRuns(d);
  if(d.system){ try{ renderRes(d.system); }catch(e){ jsErr('资源渲染失败: '+e.message); } }
  if(!d.ok){ conn.textContent='✕ '+(d.error||'读取失败'); conn.className='meta dead'; return; }
  $('clock').textContent='服务器时间 '+d.server_time;

  const cur=d.current||{};
  const done=d.done_count||0, tot=d.total_epochs||'?';
  $('epochline').textContent=`已完成 ${done} / ${tot} 轮`;
  $('epbar').style.width=(tot&&tot>0)?Math.min(100,done*100/tot)+'%':'0%';
  const pp=cur.progress_pct;
  // 两个进度条的关系要说清，否则看着像『轮次里还有轮次』：
  //   上面 = 第几轮 / 共几轮
  //   下面 = 这一轮内部跑到第几批
  // 现在 1 轮 = 1/5 遍数据（4,309 批），所以下面这条会反复跑满，属正常。
  if($('stepnote')) $('stepnote').textContent='  （这一轮内部的批次进度，共 4,309 批；跑满即进入下一轮）';
  $('stepline').textContent=(cur.epoch!==undefined)
    ? `第 ${cur.epoch} 轮`+(pp!==null&&pp!==undefined?` · ${pp}%`:'')
      +(cur.progress_loss!==null&&cur.progress_loss!==undefined?` · 实时损失 ${fmt(cur.progress_loss,4)}`:'')
    : '—';
  $('stepbar').style.width=(pp||0)+'%';

  $('tloss').textContent=fmt(cur.train_loss);
  $('vloss').textContent=fmt(cur.val_loss);
  $('aucp').textContent=fmt(cur.auc_pooled);
  $('aucd').textContent=fmt(cur.auc_daily);
  $('ic').textContent=sgn(cur.ic);
  $('icir').textContent=sgn(cur.icir,3);
  $('ict').textContent=sgn(cur.ic_t,2);
  $('hit').textContent=fmt(cur.topk_hit);
  $('topret').textContent=(cur.top_return!==undefined&&cur.top_return!==null)?((cur.top_return>=0?'+':'')+fmt(cur.top_return,3)+'%'):'—';
  $('realret').textContent=(cur.real_return!==undefined&&cur.real_return!==null)?((cur.real_return>=0?'+':'')+fmt(cur.real_return,2)+'%'):'—';
  $('lr').textContent=(cur.lr!==undefined&&cur.lr!==null)?cur.lr.toExponential(3):'—';
  $('note').textContent=cur.note||'—';
  $('days').textContent=cur.eval_days!==undefined?cur.eval_days:'—';
  $('nep').textContent=done;

  const bl=d.baseline;
  $('baseline').textContent=bl?(bl.file||'已找到'):'未找到基线档案';
  const bests=d.best||[];
  $('best').innerHTML=bests.length?bests.slice(-14).reverse()
      .map(b=>`Epoch ${b.epoch} · ${b.what}`).join('<br>'):'—';

  STATE.epochs=d.epochs||[];   STATE.compare=d.compare||[];
  renderExtras(d);
  STATE.steps=d.step_series||[];
  if($('stepsrc')) $('stepsrc').textContent =
      d.step_source==='csv' ? '（来自 step_metrics.csv）'
    : d.step_source==='log' ? '（没找到 step_metrics.csv，已用日志里的逐批损失重建）'
    : '（暂时没有逐步损失数据）';
  updateStepChart();
  STATE.res=d.res_hist||[];
  updateChart('bigchart');
  if(charts['reschart']) updateChart('reschart');
  if(document.body.classList.contains('fs')&&charts['fschart']) updateChart('fschart');
  $('tail').textContent=(d.tail||[]).join('\n');
}

async function tick(){
  try{
    const q = STATE.run ? ('?run='+encodeURIComponent(STATE.run)) : '';
    const r=await fetch('api/status'+q,{cache:'no-store'});
    render(await r.json());
  }catch(e){
    $('conn').textContent='✕ 无法连接服务: '+e; $('conn').className='meta dead';
  }
}

function jsErr(msg){
  const el=document.getElementById('jserr');
  if(el){ el.style.display='block'; el.textContent=String(msg); }
}

// ---- 启动 ----
window.addEventListener('error', e => jsErr('JS 错误: '+(e.message||e)+
  (e.lineno?('  @行 '+e.lineno):'')));
window.addEventListener('unhandledrejection', e => jsErr('未处理的 Promise 异常: '+e.reason));

function boot(){
  try{
    if(typeof echarts === 'undefined')
      throw new Error('ECharts 未加载：static/echarts.min.js 取不到');
    initChart('bigchart', curMetric, false);
    initChart('reschart', 'res', false);   // 与训练曲线同高，用常规边距而非紧凑布局
    if(charts['reschart']) charts['reschart'].__metric='res';
    const sce=document.getElementById('stepchart');
    if(sce) charts['stepchart']=echarts.init(sce, null, {renderer:'canvas'});
    buildButtons('metricBtns', ()=>{ if(charts['bigchart']) charts['bigchart'].__metric=curMetric; updateChart('bigchart'); refreshButtons(); });
    buildButtons('metricBtnsFs', ()=>{ if(charts['fschart']) charts['fschart'].__metric=curMetric; updateChart('fschart'); refreshButtons(); });
    const bc=document.querySelector('#bigchart');
    if(bc && bc.parentElement) bc.parentElement.addEventListener('dblclick', ()=>toggleFs(true));
    window.addEventListener('resize', ()=>{ Object.values(charts).forEach(c=>c&&c.resize()); });
  }catch(e){
    jsErr('图表初始化失败（数据仍会刷新）: '+e.message);
  }
  // 无论图表是否成功，数据轮询都必须跑起来
  tick();
  setInterval(tick, 2500);
}
boot();
</script>
</body>
</html>
"""



def _sanitize_json(o):
    """把 NaN / +Inf / -Inf 递归替换成 None。

    ## 为什么必须做这一步

    Python 的 json.dumps 默认会写出裸的 `NaN` / `Infinity` —— 它们【不是合法 JSON】
    （RFC 8259 只认 null/true/false/数字/字符串/数组/对象）。
    Python 自己的 json.loads 容忍它们，所以用 Python 自检时一切正常；
    但浏览器的 JSON.parse 会直接抛
        SyntaxError: Unexpected token 'N', ..."ic": NaN... is not valid JSON
    页面就此停止刷新（d512 轮遇到的就是这个）。

    ⚠ 必须定义在【模块级】，不能塞进 Handler 类里 —— 类体被打断会让
      do_GET 等后续方法凭空消失，服务返回 501 Unsupported method ('GET')。
    """
    if isinstance(o, float):
        return None if (o != o or o in (float('inf'), float('-inf'))) else o
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    return o


# ========================= 逐步指标 =========================

def resolve_steps_path(path):
    """--steps 可以给文件，也可以给 out 目录（自动取最新的 step_metrics.csv）"""
    if not path:
        return None
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "*", "step_metrics.csv")),
                      key=os.path.getmtime)
        return hits[-1] if hits else None
    return path


def load_step_series(path, max_points=1500):
    """读取 train.py 写出的 step_metrics.csv，返回 [[step, train_loss], ...]"""
    if not path or not os.path.isfile(path):
        return []
    pts = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    pts.append([int(row[0]), float(row[3])])
                except ValueError:
                    continue
    except OSError:
        return []
    if len(pts) > max_points:
        stride = len(pts) // max_points + 1
        pts = pts[::stride]
    return pts


class Handler(BaseHTTPRequestHandler):
    explicit_log = None          # 只在 --log 显式指定时非空；为空即自动发现
    runs_dir = DEFAULT_RUNS_DIR
    out_dir = DEFAULT_OUT_DIR
    steps_path = None
    results_path = None

    def _pick_target(self):
        """解析出这次请求要看哪份日志、哪份逐步指标。

        ?run=<文件名> 可以在页面上临时切到历史运行（不重启服务）；
        不传就按 current_run.txt → 最新日志自动跟随。
        """
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        want = (q.get('run') or [''])[0]
        log = None
        if want and want != '__auto__' and RE_LOG_STAMP.search(want):
            cand = os.path.join(self.runs_dir, os.path.basename(want))
            if os.path.isfile(cand) and looks_like_training_log(cand):
                log = cand
        if log is None:
            log = resolve_log(self.explicit_log, self.runs_dir)
        steps = resolve_steps_path(self.steps_path) or find_step_csv(self.out_dir, log)
        return log, steps

    def _send(self, code, body, ctype):
        data = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith('/api/status'):
            try:
                log, steps = self._pick_target()
                d = parse_log(log)
                # 评估结果来自独立的 _eval_daemon.py，不在训练日志里
                merge_eval_results(d, self.results_path)
                # 消融对照：把其它 run 的曲线一并返回，前端叠加显示
                d['compare'] = getattr(self, '_compare_cache', None) or []
                d['system'] = collect_system()
                d['res_hist'] = res_history()
                series = load_step_series(steps)
                source = 'csv'
                if not series:
                    series = d.pop('log_step_series', None) or []
                    source = 'log' if series else 'none'
                else:
                    d.pop('log_step_series', None)
                d['step_series'] = series
                d['step_source'] = source
                d['steps_csv'] = steps
                d['auto'] = self.explicit_log is None
                d['runs'] = [{'name': r['name'], 'mtime': r['mtime']}
                             for r in list_runs(self.runs_dir)]
                self._send(200, json.dumps(_sanitize_json(d), ensure_ascii=False),
                           'application/json; charset=utf-8')
            except Exception as e:
                self._send(500, json.dumps(_sanitize_json({'ok': False, 'error': repr(e)}),
                                           ensure_ascii=False),
                           'application/json; charset=utf-8')
        elif self.path.startswith('/static/'):
            name = os.path.basename(self.path[len('/static/'):])
            fp = os.path.join(STATIC_DIR, name)
            if os.path.isfile(fp):
                ctype = ('application/javascript' if name.endswith('.js')
                         else 'text/css' if name.endswith('.css')
                         else 'application/octet-stream')
                with open(fp, 'rb') as f:
                    self._send(200, f.read(), ctype)
            else:
                self._send(404, 'not found', 'text/plain; charset=utf-8')
        elif self.path in ('/', '/index.html'):
            self._send(200, HTML, 'text/html; charset=utf-8')
        else:
            self._send(404, 'not found', 'text/plain; charset=utf-8')

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(
        description='训练监控服务。默认自动跟随 train_logs/current_run.txt，'
                    '换一次训练不需要改任何参数。')
    ap.add_argument('--log', default=None,
                    help='固定只看这一份日志；不给则自动跟随 current_run.txt / 最新日志')
    ap.add_argument('--runs', default=DEFAULT_RUNS_DIR,
                    help=f'训练日志目录，页面上的运行切换列表来自这里（默认 {DEFAULT_RUNS_DIR}）')
    ap.add_argument('--out', default=DEFAULT_OUT_DIR,
                    help=f'训练输出目录，用于自动匹配 step_metrics.csv（默认 {DEFAULT_OUT_DIR}）')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--static', default=None,
                    help='echarts.min.js 所在目录，默认 <仓库根>/static')
    ap.add_argument('--compare', action='append', default=[],
                    help='对照实验曲线，格式 标签=路径，可重复。'
                         '用于消融对比，会以虚线叠加到各图表上')
    ap.add_argument('--steps', default=None,
                    help='固定用这份 step_metrics.csv；不给则按日志时间自动匹配')
    ap.add_argument('--results', default=None,
                    help='_eval_daemon.py 写出的 eval_results.jsonl 路径')
    args = ap.parse_args()

    global STATIC_DIR
    if args.static:
        STATIC_DIR = args.static
    Handler.explicit_log = args.log
    Handler.runs_dir = args.runs
    Handler.out_dir = args.out
    Handler.steps_path = args.steps
    Handler.results_path = args.results

    # 预加载对照曲线（只做一次，之后每次请求复用）
    _specs = []
    for item in (args.compare or []):
        if '=' not in item:
            print(f'  ⚠ --compare 格式应为 标签=路径，忽略: {item}')
            continue
        lb, pth = item.split('=', 1)
        _specs.append((lb.strip(), pth.strip()))
    Handler._compare_cache = load_compare_series(_specs)
    if not os.path.isfile(os.path.join(STATIC_DIR, 'echarts.min.js')):
        print(f"⚠ 未找到 {STATIC_DIR}/echarts.min.js —— 图表将无法渲染")

    stop = threading.Event()
    th = threading.Thread(target=_sampler_loop, args=(stop,), daemon=True)
    th.start()
    print(f"资源采样线程已启动（每 {SAMPLE_INTERVAL}s，图上窗口 {RES_WINDOW_SEC}s）")

    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"监控服务: http://{args.bind}:{args.port}/")
    if Handler.explicit_log:
        print(f"日志: {Handler.explicit_log}（--log 固定）")
        print(f"逐步指标: {resolve_steps_path(Handler.steps_path) or find_step_csv(Handler.out_dir, Handler.explicit_log)}")
    else:
        log = resolve_log(None, Handler.runs_dir)
        print(f"日志: {log or '（暂未发现）'}（自动跟随 {Handler.runs_dir}）")
        print(f"逐步指标: {find_step_csv(Handler.out_dir, log) or '（自动匹配）'}")
    try:
        srv.serve_forever()
    finally:
        stop.set()


if __name__ == '__main__':
    main()


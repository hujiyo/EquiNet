#!/usr/bin/env bash
# 启动训练监控页。不带任何参数：自动跟随 train_logs/current_run.txt 指向的训练，
# 逐步指标按日志时间戳自动匹配 out/<运行目录>/step_metrics.csv。
#
# 用法:
#   tools/start_monitor.sh            # 前台运行，端口 8080
#   tools/start_monitor.sh 8081       # 换端口
#   nohup tools/start_monitor.sh &    # 后台运行
#
# 需要钉死看某一份文件时，直接调 python tools/train_monitor.py --log ... --steps ...
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8080}"
RUNS_DIR="${RUNS_DIR:-$(cd .. && pwd)/train_logs}"

# 非交互式 ssh 不会自动激活 conda 环境，这里自己激活；激活不了就用系统 python。
if [ -z "${CONDA_PREFIX:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate equinet
fi
PY="${PYTHON:-python}"

mkdir -p "$RUNS_DIR/monitors"
exec "$PY" -u tools/train_monitor.py --port "$PORT" \
    >> "$RUNS_DIR/monitors/monitor_$(date +%Y%m%d_%H%M%S).log" 2>&1

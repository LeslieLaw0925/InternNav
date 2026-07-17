#!/bin/bash

# ================= 配置参数 =================
DEV=eno1            # 网卡接口名称
DOWN_TIME=5         # 每次断网持续时间（秒）
UP_TIME=15          # 每次网络正常维持时间（秒）
# ============================================

# 安全清理函数：确保不论发生什么中断，网卡规则都能被安全擦除
cleanup() {
    echo -e "\n\n[!] 捕获到终止信号，正在强行恢复网络..."
    sudo tc qdisc del dev $DEV root 2>/dev/null
    echo "[+] 网络已完全恢复正常，退出脚本。"
    exit 0
}

# 绑定 Ctrl+C (SIGINT) 和 终止 (SIGTERM) 信号到清理函数
trap cleanup SIGINT SIGTERM

echo "============================================="
echo "   开始周期性断网仿真测试 (系统仿真器专用)   "
echo "   目标网卡: $DEV"
echo "   断网时长: ${DOWN_TIME}s  |  正常时长: ${UP_TIME}s"
echo "   提示: 随时按下 [Ctrl+C] 可安全停止并恢复网络"
echo "============================================="

# 周期循环
CYCLE_NUM=1
while true; do
    echo "---------------------------------------------"
    echo "[$(date '+%H:%M:%S')] -> 第 #$CYCLE_NUM 次断网循环开始..."
    
    # 1. 注入 100% 丢包
    echo "[DOWN] 开始断网 (loss 100%)..."
    sudo tc qdisc add dev $DEV root netem loss 100%
    sleep $DOWN_TIME

    # 2. 擦除注入规则恢复网络
    echo "[UP]   网络恢复正常..."
    sudo tc qdisc del dev $DEV root 2>/dev/null
    
    echo "[WAIT] 维持正常网络环境中..."
    sleep $UP_TIME

    CYCLE_NUM=$((CYCLE_NUM + 1))
done
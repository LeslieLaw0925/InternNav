#!/bin/bash

# --- 配置區 ---
IFACE="eno1"        # 你的網卡名稱
INTERVAL=5          # 每筆數據的切換間隔（秒）

MIN_BW=1                  # 極低帶寬下限 (1 Mbps)
MAX_BW=5
# --------------

stop_limit() {
    tc qdisc del dev $IFACE root 2>/dev/null
    echo "已清除 $IFACE 的限制"
}

start_replay() {
    stop_limit
    
    # 建立初始 qdisc
    # 使用 tbf (Token Bucket Filter) 進行限速
    # burst 建議根據最大頻寬調整，這裡設為 32k
    tc qdisc add dev $IFACE root handle 1: tbf rate 20mbit burst 32k latency 400ms

    # 按行讀取頻寬數據
    COUNT=0
    while (( COUNT >= 0 )); do
        ((COUNT++))
        # 實時修改頻寬限制
        BW=$(( RANDOM % (MAX_BW - MIN_BW + 1) + MIN_BW ))
        tc qdisc change dev $IFACE root handle 1: tbf rate ${BW}mbit burst 32k latency 400ms
        
        echo -ne "网络数据_${COUNT} | 当前带宽: ${BW} Mbps   \r"
        
        sleep $INTERVAL
    done

    echo -e "\n✅ 网络数据回放结束。"
    stop_limit
}

# 捕獲 Ctrl+C，確保退出時清除限制
trap 'echo -e "\n中止回放"; stop_limit; exit' SIGINT SIGTERM

case "$1" in
    start)  start_replay ;;
    stop)   stop_limit ;;
    *)      echo "用法: sudo $0 {start|stop}" ;;
esac
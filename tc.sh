#!/bin/bash

# --- 配置區 ---
IFACE="enp7s0"        # 你的網卡名稱
TRACE_FILE="4g_trace.txt"
INTERVAL=10          # 每筆數據的切換間隔（秒）
# --------------

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "找不到 $TRACE_FILE, 请确保文件存在"
    exit 1
fi

stop_limit() {
    tc qdisc del dev $IFACE root 2>/dev/null
    echo "已清除 $IFACE 的限制"
}

start_replay() {
    stop_limit
    echo "🚀 开始回放真实网络数据: $TRACE_FILE"
    
    # 建立初始 qdisc
    # 使用 tbf (Token Bucket Filter) 進行限速
    # burst 建議根據最大頻寬調整，這裡設為 32k
    tc qdisc add dev $IFACE root handle 1: tbf rate 20mbit burst 32k latency 400ms

    ROUND=0
    while true; do
        ((ROUND++))
        echo -e "\n🔄 开始第 ${ROUND} 轮回放"
        
        # 按行讀取頻寬數據
        COUNT=0
        while IFS= read -r BW; do
            ((COUNT++))
            # 實時修改頻寬限制
            tc qdisc change dev $IFACE root handle 1: tbf rate ${BW}mbit burst 32k latency 400ms
            
            echo -ne "网络数据_${COUNT} | 当前带宽: ${BW} Mbps   \r"
            
            sleep $INTERVAL
        done < "$TRACE_FILE"
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
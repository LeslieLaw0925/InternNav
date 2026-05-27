#!/bin/bash

# --- 配置區 ---
IFACE="enp7s0"            # 你的網卡名稱
TRACE_FILE="4g_trace.txt"
INTERVAL=100              # 每筆數據的切換間隔（秒）
TARGET_IP="192.168.105.11" # S2的 IP
BASE_RATE="1000mbit"
# --------------

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "找不到 $TRACE_FILE, 请确保文件存在"
    exit 1
fi

stop_limit() {
    # 刪除根節點，會連同下屬分類和過濾器一併清除
    tc qdisc del dev $IFACE root 2>/dev/null
    echo "已清除 $IFACE 對 $TARGET_IP 的限制"
}

start_replay() {
    stop_limit
    echo "🚀 開始回放真實網路數據至 $TARGET_IP: $TRACE_FILE"
    
    # 1. 在網卡上建立一個根隊列 (root qdisc)
    # default 1 表示未匹配到任何 filter 的流量走 classid 1:1（無限制類）
    tc qdisc add dev $IFACE root handle 1: htb default 1

    # 2. 建立主要類別 (class 1:1) - 這是默認類，用於所有其他 IP 的流量（無限制）
    tc class add dev $IFACE parent 1: classid 1:1 htb rate $BASE_RATE

    # 3. 建立受限類別 (class 1:2) - 專門用於限制目標 IP 的流量
    # 初始帶寬設為 20mbit，後續會根據數據文件動態修改
    tc class add dev $IFACE parent 1: classid 1:2 htb rate 20mbit ceil 20mbit

    # 4. 建立過濾器 (filter)，將目標 IP 的流量導向受限類 1:2
    # 注意：這裡限制的是從這台機器「流向」TARGET_IP 的流量 (Egress)
    tc filter add dev $IFACE protocol ip parent 1: prio 1 u32 match ip dst $TARGET_IP flowid 1:2

    # 按行讀取頻寬數據
    COUNT=0
    while IFS= read -r BW; do
        ((COUNT++))
        
        # 4. 實時修改該類別的頻寬限制 (使用 change 命令)
        tc class change dev $IFACE parent 1: classid 1:2 htb rate ${BW}mbit ceil ${BW}mbit
        
        echo -ne "網絡數據_${COUNT} | 對目標 ${TARGET_IP} 當前帶寬: ${BW} Mbps   \r"
        
        sleep $INTERVAL
    done < "$TRACE_FILE"

    echo -e "\n✅ 網絡數據回放結束。"
    stop_limit
}

# 捕獲 Ctrl+C
trap 'echo -e "\n中止回放"; stop_limit; exit' SIGINT SIGTERM

case "$1" in
    start)  start_replay ;;
    stop)   stop_limit ;;
    *)      echo "用法: sudo $0 {start|stop}" ;;
esac
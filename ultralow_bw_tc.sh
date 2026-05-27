#!/bin/bash

# --- 配置區 ---
IFACE="enp7s0"            # 你的網卡名稱
INTERVAL=5              # 每筆數據的切換間隔（秒）
TARGET_IP="192.168.105.11" # S2的 IP
BASE_RATE="1000mbit"

MIN_BW=1                  # 極低帶寬下限 (1 Mbps)
MAX_BW=5                  # 極低帶寬上限 (5 Mbps)
# --------------

stop_limit() {
    # 刪除根節點，會連同下屬分類和過濾器一併清除
    tc qdisc del dev $IFACE root 2>/dev/null
    echo "已清除 $IFACE 對 $TARGET_IP 的限制"
}

start_replay() {
    stop_limit
    
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
    while (( COUNT >= 0 )); do
        ((COUNT++))
        # 使用 Bash 的 $RANDOM 生成 1 到 5 之間的隨機整數
        BW=$(( RANDOM % (MAX_BW - MIN_BW + 1) + MIN_BW ))

        # 4. 實時修改該類別的頻寬限制 (使用 change 命令)
        tc class change dev $IFACE parent 1: classid 1:2 htb rate ${BW}mbit ceil ${BW}mbit
        
        echo -ne "對目標 ${TARGET_IP} 當前帶寬: ${BW} Mbps   \r"
        
        sleep $INTERVAL
    done

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
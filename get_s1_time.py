import json

file_path = "/home/data/projects/InternNav/logs/system_perf.jsonl"  # 替换成你的文件路径

s1_times = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "s1_time" in data:
            s1_times.append(data["s1_time"])

if s1_times:
    s1_times = s1_times[1:] 
    avg_s1_time = sum(s1_times) / len(s1_times)
    print(f"共读取 {len(s1_times)} 条数据")
    print(f"平均 s1_time: {avg_s1_time:.6f} s")
else:
    print("没有找到任何 s1_time 数据")
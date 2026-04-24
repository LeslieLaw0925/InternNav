# extract_trace.py
filename = "lateral.list"
target_run = "Run 4"  # 你可以指定要回放哪一次的數據
min_bw = 5
offset = 20

with open(filename, "r") as f:
    lines = f.read().split("###############################")
    for section in lines:
        if target_run in section:
            # 提取數字部分
            data_str = section.split(target_run)[1].strip()
            # 轉換為每行一個頻寬值
            with open("4g_trace.txt", "w") as out:
                for val in data_str.split(","):
                    if val.strip():
                        val = val.strip()
                        modified_val = max(min_bw, float(val) - offset)
                        out.write(f"{modified_val}\n")
            print(f"✅ 已成功提取 {target_run} 數據到 4g_trace.txt")
            break
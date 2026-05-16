import pandas as pd
import json

# 假设你的文件名是 data.json
file_path = 'progress.json'

def calculate_json_means(path):
    data = []
    
    # 尝试以 JSON Lines 格式读取（每行一个对象）
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
    except Exception:
        # 如果上面失败，尝试直接作为标准 JSON 列表读取
        df = pd.read_json(path)

    # 筛选出数值类型的列（排除 scene_id 等字符串）
    numeric_df = df.select_dtypes(include=['number'])
    
    # 计算各列均值
    means = numeric_df.mean()
    
    print("--- 各项指标均值结果 ---")
    print(means)
    return means

# 运行
results = calculate_json_means(file_path)
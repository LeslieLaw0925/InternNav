import re

# 匹配示例：
# Cloud inference time: 123.45 ms
# Cloud inference time = 0.56s
PATTERN = re.compile(
    r"Cloud inference time\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Zμµ]*)",
    re.IGNORECASE,
)

s1_pattern = re.compile(
    r"On-device system1 step time\s*[:=]\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Zμµ]*)",
    re.IGNORECASE,
)

def extract_cloud_inference_times(log_path: str):
    values, s1_times = [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                num = float(m.group(1))
                values.append(num)

            m1 = s1_pattern.search(line)
            if m1:
                num1 = float(m1.group(1))
                s1_times.append(num1)

    return values, s1_times


if __name__ == "__main__":
    log_file = "nextdit_bl.log"
    results, s1_results = extract_cloud_inference_times(log_file)
    # print(f'On-device system1 step times: {s1_results}')

    min_time = []
    med_time = []
    max_time = []

    # 仅输出提取到的值（每行一个）
    for v in s1_results:
        if v < 1.3:
            min_time.append(v)
        elif v < 1.5:
            med_time.append(v)
        else:
            max_time.append(v)

    # print(f'Min inference times: {len(min_time)}')
    # print(f'Medium inference times: {len(med_time)}')
    # print(f'Max inference times: {len(max_time)}')

    print(f'Mean time of min inference times: {sum(min_time) / len(min_time) if min_time else 0}')
    print(f'Mean time of max inference times: {sum(max_time) / len(max_time) if max_time else 0}')

    # print(f'Total number of cloud inference times extracted: {len(results)}')
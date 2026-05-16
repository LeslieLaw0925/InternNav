import json


class InferenceLogger:
    def __init__(self, log_path="logs/system_perf.jsonl"):
        self.log_path = log_path
        self.entry = {}

    def record_by_key(self, key: str, value: float):
        self.entry[key] = value
       
    def flush(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.entry) + "\n")
            f.flush()

        self.reset()

    def reset(self):
        self.entry = {}
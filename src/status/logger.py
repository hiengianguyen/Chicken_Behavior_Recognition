import json
import os


class StatusLogger:

    def __init__(self, log_path="logs/chicken_status.jsonl"):

        self.log_path = log_path

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_snapshot(self, snapshot):

        with open(self.log_path, "a", encoding="utf-8") as file:

            file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

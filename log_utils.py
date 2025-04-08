import os
import csv
from datetime import datetime

def save_keyword_log(keywords: list[str]):
    log_path = "logs/keyword_log.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().isoformat()
        row = [timestamp] + keywords[:5]
        writer.writerow(row)
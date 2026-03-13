"""
Session logger — writes one CSV row per frame to data/sessions/.

Each session gets its own timestamped file so runs never overwrite each other.
The CSV is the raw material for post-session analysis (feature engineering,
drowsiness summaries, etc.).

Columns:
  timestamp    Unix time (seconds, 3 dp)
  frame        Frame counter
  ear          Eye Aspect Ratio (averaged across both eyes)
  mar          Mouth Aspect Ratio
  perclos      Fraction of recent frames with eyes closed  (0-1)
  blink_rate   Estimated blinks per minute
  alert_level  NORMAL / LOW / MEDIUM / HIGH / CRITICAL
"""
import csv
from pathlib import Path
from datetime import datetime


class SessionLogger:
    COLUMNS = ["timestamp", "frame", "ear", "mar", "perclos", "blink_rate", "alert_level"]

    def __init__(self, log_dir: str = "data/sessions"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        filename  = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
        self.path = Path(log_dir) / filename

        self._file   = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def log(
        self,
        frame: int,
        ear: float,
        mar: float,
        perclos: float,
        blink_rate: float,
        alert_level: str,
    ):
        self._writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame,
            round(ear,        4),
            round(mar,        4),
            round(perclos,    4),
            round(blink_rate, 2),
            alert_level,
        ])

    def close(self):
        self._file.close()
        print(f"📁 Session saved → {self.path}")
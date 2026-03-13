"""Session logger — records per-frame metrics to a CSV file"""
import csv
import time
from datetime import datetime
from pathlib import Path


class SessionLogger:
    """Logs one CSV row per processed frame for post-session analysis"""

    HEADER = [
        'timestamp',
        'frame_num',
        'ear',
        'mar',
        'perclos',
        'blink_rate',
        'alert_level',
        'consecutive_drowsy_frames',
        'microsleep_flag',
    ]

    def __init__(self):
        # Create data/logs/ directory if it doesn't exist
        log_dir = Path('data/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped filename
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = log_dir / f'session_{ts}.csv'

        self._file = open(self.csv_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log_frame(self, frame_num, metrics, alert_level, consecutive_drowsy):
        """Append one row for the current frame.

        Args:
            frame_num: integer frame counter
            metrics: dict with keys ear, mar, perclos, blink_rate
            alert_level: string e.g. "NORMAL", "HIGH", "CRITICAL"
            consecutive_drowsy: int number of consecutive drowsy frames
        """
        microsleep_flag = 1 if alert_level == 'CRITICAL' else 0
        self._writer.writerow([
            round(time.time(), 4),
            frame_num,
            round(metrics.get('ear', 0.0), 4),
            round(metrics.get('mar', 0.0), 4),
            round(metrics.get('perclos', 0.0), 4),
            round(metrics.get('blink_rate', 0.0), 2),
            alert_level,
            consecutive_drowsy,
            microsleep_flag,
        ])

    def finalize(self):
        """Close the CSV file and return its path.

        Returns:
            pathlib.Path pointing to the written CSV
        """
        self._file.close()
        return self.csv_path

    def __del__(self):
        """Ensure the file is closed even if finalize() was never called."""
        if not self._file.closed:
            self._file.close()

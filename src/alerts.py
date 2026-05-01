"""Alert system with cooldown."""
import time


class AlertSystem:
    """Simple alert system with cooldown."""

    def __init__(self, cooldown_seconds=3):
        self.cooldown = cooldown_seconds
        self.last_alert_time = 0

    def should_alert(self):
        """Return True when enough time has passed since the last alert."""
        current_time = time.time()
        return current_time - self.last_alert_time >= self.cooldown

    def trigger(self, level, metrics):
        """Print an alert if the cooldown has elapsed."""
        if not self.should_alert():
            return False

        self.last_alert_time = time.time()

        print(
            f"\n! {level} ALERT - "
            f"EAR: {metrics.get('ear', 0):.3f} | "
            f"PERCLOS: {metrics.get('perclos', 0):.2%} | "
            f"Score: {metrics.get('score', 0):.1f} | "
            f"Microsleep: {metrics.get('microsleep', 0):.2f}s"
        )

        return True

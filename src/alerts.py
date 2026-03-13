"""Alert system"""
import time


class AlertSystem:
    """Simple alert system with cooldown"""
    
    def __init__(self, cooldown_seconds=3):
        self.cooldown = cooldown_seconds
        self.last_alert_time = 0

    def play_alert_sound(self, level):
        """
        Play a beep pattern based on alert level.

        HIGH     - triple beep (1500 Hz, 400 ms each)
        CRITICAL - rapid 5 beeps (2000 Hz, 150 ms each)

        Fails silently on non-Windows platforms where winsound is unavailable.
        """
        try:
            import winsound
            if level == "HIGH":
                for _ in range(3):
                    winsound.Beep(1500, 400)
            elif level == "CRITICAL":
                for _ in range(5):
                    winsound.Beep(2000, 150)
        except Exception:
            pass

    def should_alert(self):
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown:
            return True
        return False
    
    def trigger(self, level, metrics):
        """
        Trigger alert
        
        Args:
            level: "LOW", "MEDIUM", "HIGH", "CRITICAL"
            metrics: dict with ear, perclos, etc.
        """
        if not self.should_alert():
            return False
        
        # Update last alert time
        self.last_alert_time = time.time()
        
        # Print alert
        symbols = {
            "LOW": "⚡",
            "MEDIUM": "⚠️",
            "HIGH": "🚨",
            "CRITICAL": "🔴"
        }
        
        symbol = symbols.get(level, "⚡")
        print(f"\n{symbol} {level} ALERT - "
              f"EAR: {metrics.get('ear', 0):.3f} | "
              f"PERCLOS: {metrics.get('perclos', 0):.2%}")
        
        self.play_alert_sound(level)
        
        return True
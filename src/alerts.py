"""Alert system"""
import time


class AlertSystem:
    """Simple alert system with cooldown"""
    
    def __init__(self, cooldown_seconds=3):
        self.cooldown = cooldown_seconds
        self.last_alert_time = 0
        
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
        
        # Optional: Add sound here later
        # import winsound
        # winsound.Beep(1000, 500)
        
        return True
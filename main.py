"""
Driver Drowsiness Detection System - Laptop Demo
"""
import cv2
import yaml
import time
from pathlib import Path

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DrowsinessMetrics
from src.alerts import AlertSystem
from src.logger import SessionLogger
from src.session_summary import generate_summary


class DrowsinessDetectionSystem:
    """Main application"""
    
    def __init__(self):
        # Load config
        with open('config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.camera = Camera(
            source=self.config['camera']['source'],
            width=self.config['camera']['width'],
            height=self.config['camera']['height'],
            fps=self.config['camera']['fps']
        )
        
        self.detector = FaceDetector(
            min_detection_confidence=self.config['detection']['min_detection_confidence'],
            min_tracking_confidence=self.config['detection']['min_tracking_confidence']
        )
        
        fps = self.config['camera']['fps']
        perclos_seconds = self.config['drowsiness']['perclos_window_seconds']
        
        self.metrics = DrowsinessMetrics(
            ear_threshold=self.config['drowsiness']['ear_threshold'],
            perclos_window_frames=int(fps * perclos_seconds)
        )
        
        self.alerts = AlertSystem(
            cooldown_seconds=self.config['alerts']['cooldown_seconds']
        )
        
        # State tracking
        self.consecutive_drowsy = 0
        self.consecutive_yawn = 0
        self.flash_start_time = 0
        self.frame_num = 0
        
        # Session logger
        self.logger = SessionLogger()
        
        # Thresholds
        self.ear_threshold = self.config['drowsiness']['ear_threshold']
        self.ear_frames = self.config['drowsiness']['ear_consecutive_frames']
        self.mar_threshold = self.config['drowsiness']['mar_threshold']
        self.mar_frames = self.config['drowsiness']['mar_consecutive_frames']
        self.perclos_threshold = self.config['drowsiness']['perclos_threshold']
        
    def determine_alert_level(self, ear, mar, perclos, blink_rate):
        """Determine drowsiness severity"""
        
        # Critical: Multiple indicators or severe drowsiness
        if perclos > 0.3 or self.consecutive_drowsy > 60:
            level = "CRITICAL"
        # High: PERCLOS above threshold or prolonged eye closure
        elif perclos > self.perclos_threshold or self.consecutive_drowsy > self.ear_frames:
            level = "HIGH"
        # Medium: Yawning detected
        elif self.consecutive_yawn > self.mar_frames:
            level = "MEDIUM"
        # Low: Brief eye closure
        elif self.consecutive_drowsy > 10:
            level = "LOW"
        else:
            level = "NORMAL"

        # Upgrade level by one step when blink_rate signals drowsiness
        if blink_rate != 0 and 1 <= blink_rate <= 10 and perclos > 0.10:
            upgrade = {"NORMAL": "LOW", "LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "CRITICAL"}
            level = upgrade.get(level, level)

        return level
    
    def draw_ui(self, frame, metrics, alert_level, fps):
        """Draw UI overlay"""
        
        # Color based on alert level
        colors = {
            "NORMAL": (0, 255, 0),
            "LOW": (0, 255, 255),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 100, 255),
            "CRITICAL": (0, 0, 255)
        }
        color = colors.get(alert_level, (0, 255, 0))

        # Full-screen color flash for HIGH and CRITICAL (visible for 0.5 s)
        if alert_level in ("HIGH", "CRITICAL"):
            elapsed = time.time() - self.flash_start_time
            if elapsed < 0.5:
                flash_color = (0, 0, 255) if alert_level == "CRITICAL" else (0, 165, 255)
                flash_overlay = frame.copy()
                cv2.rectangle(flash_overlay, (0, 0),
                              (frame.shape[1], frame.shape[0]),
                              flash_color, -1)
                cv2.addWeighted(flash_overlay, 0.3, frame, 0.7, 0, frame)
        
        # Dark overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Alert status
        cv2.putText(frame, f"Status: {alert_level}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Metrics
        cv2.putText(frame, f"EAR: {metrics['ear']:.3f}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"PERCLOS: {metrics['perclos']:.1%}", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Blink Rate: {metrics['blink_rate']:.1f}/min", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Drowsiness bar
        bar_width = int(metrics['perclos'] * 600)
        cv2.rectangle(frame, (300, 15), (900, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (300, 15), (300 + bar_width, 40), color, -1)
        cv2.putText(frame, "Drowsiness", (310, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        
        print("🚗 Driver Drowsiness Detection System")
        print("=" * 50)
        print("Platform: Laptop (Windows)")
        print("GPU: GTX 1650Ti")
        print("Camera: 720p @ 30fps")
        print("Detection: MediaPipe Face Mesh")
        print("=" * 50)
        
        try:
            self.camera.start()
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return
        
        print("✅ System ready")
        print("📹 Press 'q' to quit\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️ Failed to read frame")
                    continue
                
                # Detect face
                landmarks = self.detector.detect(frame)
                
                if landmarks is None:
                    # No face detected
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Drowsiness Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Get eyes and mouth
                left_eye, right_eye = self.detector.get_eyes(landmarks)
                mouth = self.detector.get_mouth(landmarks)
                
                # Calculate metrics
                eye_metrics = self.metrics.update(left_eye, right_eye)
                ear = eye_metrics['ear']
                perclos = eye_metrics['perclos']
                blink_rate = eye_metrics['blink_rate']
                
                mar = self.metrics.calculate_mar(mouth)
                
                # Track consecutive frames
                if ear < self.ear_threshold:
                    self.consecutive_drowsy += 1
                else:
                    self.consecutive_drowsy = 0
                
                if mar > self.mar_threshold:
                    self.consecutive_yawn += 1
                else:
                    self.consecutive_yawn = 0
                
                # Determine alert level
                alert_level = self.determine_alert_level(ear, mar, perclos, blink_rate)
                
                # Log frame metrics
                self.frame_num += 1
                self.logger.log_frame(
                    self.frame_num,
                    {'ear': ear, 'mar': mar, 'perclos': perclos, 'blink_rate': blink_rate},
                    alert_level,
                    self.consecutive_drowsy
                )
                
                # Trigger alerts
                if alert_level != "NORMAL":
                    if self.alerts.trigger(alert_level, {
                        'ear': ear,
                        'mar': mar,
                        'perclos': perclos,
                        'blink_rate': blink_rate
                    }):
                        if alert_level in ("HIGH", "CRITICAL"):
                            self.flash_start_time = time.time()
                
                # Draw landmarks
                if self.config['display']['show_landmarks']:
                    frame = self.detector.draw_landmarks(frame, landmarks)
                
                # Draw UI
                if self.config['display']['show_metrics']:
                    frame = self.draw_ui(frame, eye_metrics, alert_level, 
                                        self.camera.get_fps())
                
                # Display
                cv2.imshow('Drowsiness Detection', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏸️ Stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera.release()
        self.detector.cleanup()
        cv2.destroyAllWindows()
        
        # Finalise session log and print drive summary
        csv_path = self.logger.finalize()
        print(f"📊 Session log saved: {csv_path}")
        try:
            generate_summary(csv_path)
        except Exception as e:
            print(f"⚠️ Could not generate summary: {e}")
        
        print("✅ Shutdown complete")


if __name__ == "__main__":
    system = DrowsinessDetectionSystem()
    system.run()
"""Driver Drowsiness Detection System — Laptop Demo"""
import cv2
import yaml
from pathlib import Path

from src.camera   import Camera
from src.detector import FaceDetector
from src.metrics  import DrowsinessMetrics
from src.alerts   import AlertSystem
from src.logger   import SessionLogger


class DrowsinessDetectionSystem:

    def __init__(self):
        # Load config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        cam_cfg       = self.config.get("camera",     {})
        det_cfg       = self.config.get("detection",  {})
        alert_cfg     = self.config.get("alerts",     {})
        display_cfg   = self.config.get("display",    {})
        drowsy_cfg    = self.config.get("drowsiness", {})

        # Camera
        self.camera = Camera(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width",   1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps",       30),
            
            
        )

        # Detector
        self.detector = FaceDetector(
            min_detection_confidence=det_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=det_cfg.get("min_tracking_confidence",  0.5),
        )

        # Metrics — fps is updated after camera starts (see run())
        self.metrics = DrowsinessMetrics(
            fps=cam_cfg.get("fps", 30),
            perclos_window_seconds=drowsy_cfg.get("perclos_window_seconds", 3),
        )

        # Alerts
        self.alerts = AlertSystem(
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 3)
        )

        # Session logger
        self.logger = SessionLogger()
        self.frame_count = 0

        # Alert state
        self.consecutive_drowsy = 0
        self.consecutive_yawn   = 0

        # Display flags
        self.show_landmarks = display_cfg.get("show_landmarks", True)
        self.show_metrics   = display_cfg.get("show_metrics",   True)
        self.show_fps       = display_cfg.get("show_fps",       True)

        # Drowsiness thresholds (for alert logic)
        self.mar_threshold           = drowsy_cfg.get("mar_threshold",           0.6)
        self.ear_consecutive_frames  = drowsy_cfg.get("ear_consecutive_frames",  15)
        self.mar_consecutive_frames  = drowsy_cfg.get("mar_consecutive_frames",  15)
        self.perclos_threshold       = drowsy_cfg.get("perclos_threshold",       0.15)

    # ── calibration overlay ───────────────────────────────────────────────────

    def _draw_calibration_overlay(self, frame, current_ear):
        """
        Draws a progress banner across the top of the frame while
        the 30-second EAR baseline is being collected.
        """
        h, w    = frame.shape[:2]
        progress    = self.metrics.calibration_progress       # 0.0 → 1.0
        seconds_done = int(DrowsinessMetrics.CALIBRATION_SECONDS * progress)
        seconds_left = DrowsinessMetrics.CALIBRATION_SECONDS - seconds_done

        # Dark header band
        frame[0:115, :] = (frame[0:115, :] * 0.5).astype(frame.dtype)

        # Title
        cv2.putText(
            frame,
            "CALIBRATING  —  look straight ahead and keep eyes open",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (255, 255, 100), 2, cv2.LINE_AA,
        )

        # Progress bar background
        bar_x, bar_y, bar_h = 20, 50, 20
        bar_max_w = w - 40
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_max_w, bar_y + bar_h),
            (60, 60, 60), -1,
        )

        # Progress bar fill (green)
        fill_w = int(bar_max_w * progress)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_w, bar_y + bar_h),
                (80, 220, 130), -1,
            )

        # Border around bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_max_w, bar_y + bar_h),
            (120, 120, 120), 1,
        )

        # Percentage label inside bar
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(
            frame, pct_text,
            (bar_x + bar_max_w // 2 - 20, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Remaining time + live EAR
        samples = len(self.metrics._cal_samples)
        cv2.putText(
            frame,
            f"{seconds_left}s remaining  |  samples: {samples}  |  EAR: {current_ear:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (190, 190, 190), 1, cv2.LINE_AA,
        )

    # ── main UI overlay (normal detection mode) ───────────────────────────────

    def draw_ui(self, frame, ear, mar, perclos, blink_rate, alert_level, fps):
        h, w = frame.shape[:2]

        # Dark header band
        frame[0:140, :] = (frame[0:140, :] * 0.4).astype(frame.dtype)

        # FPS
        if self.show_fps:
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
            

        if self.show_metrics:
            # EAR
            ear_color = (0, 255, 0) if ear >= self.metrics.ear_threshold else (0, 0, 255)
            cv2.putText(
                frame, f"EAR: {ear:.3f} (thr: {self.metrics.ear_threshold:.3f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                ear_color, 2, cv2.LINE_AA,
            )

            # MAR
            mar_color = (0, 0, 255) if mar > self.mar_threshold else (0, 255, 0)
            cv2.putText(
                frame, f"MAR: {mar:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                mar_color, 2, cv2.LINE_AA,
            )

            # PERCLOS
            if perclos is not None:
                perclos_color = (0, 0, 255) if perclos > self.perclos_threshold else (0, 255, 0)
                cv2.putText(
                    frame, f"PERCLOS: {perclos:.1%}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    perclos_color, 2, cv2.LINE_AA,
                )

            # Blink rate
            if blink_rate is not None:
                cv2.putText(
                    frame, f"Blinks/min: {blink_rate:.1f}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )

        # Alert level banner (bottom)
        if alert_level and alert_level != "OK":
            colors = {
                "LOW":      (0, 255, 255),
                "MEDIUM":   (0, 165, 255),
                "HIGH":     (0, 0, 255),
                "CRITICAL": (0, 0, 200),
            }
            color = colors.get(alert_level, (255, 255, 255))
            frame[h - 60:h, :] = (frame[h - 60:h, :] * 0.4).astype(frame.dtype)
            cv2.putText(
                frame, f"⚠ {alert_level} DROWSINESS ALERT",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                color, 2, cv2.LINE_AA,
            )

        # PERCLOS progress bar
        if perclos is not None:
            bar_max_w = 600
            bar_x     = (w - bar_max_w) // 2
            bar_y     = h - 80
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 12), (50, 50, 50), -1)
            fill_w = min(int(perclos * bar_max_w), bar_max_w)
            fill_color = (0, 200, 0) if perclos < 0.15 else (0, 100, 255) if perclos < 0.30 else (0, 0, 255)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), fill_color, -1)

        # Baseline EAR reminder (small, top-right)
        if self.metrics.baseline_ear is not None:
            cv2.putText(
                frame,
                f"baseline: {self.metrics.baseline_ear:.3f}",
                (w - 210, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (120, 120, 120), 1, cv2.LINE_AA,
            )

        return frame

    # ── alert level logic ─────────────────────────────────────────────────────

    def determine_alert_level(self, perclos):
        """
        Returns "OK" / "LOW" / "MEDIUM" / "HIGH" / "CRITICAL".
        perclos=None during calibration — always returns "OK".
        """
        if perclos is None:
            return "OK"

        if perclos > 0.30 or self.consecutive_drowsy > 60:
            return "CRITICAL"
        elif perclos > 0.20 or self.consecutive_drowsy > 30:
            return "HIGH"
        elif perclos > 0.15 or self.consecutive_drowsy > 15:
            return "MEDIUM"
        elif perclos > 0.10:
            return "LOW"
        return "OK"

    # ── per-frame processing ──────────────────────────────────────────────────

    def process_frame(self, frame):
        # Detect landmarks
        landmarks = self.detector.detect(frame)

        if landmarks is None:
            # No face — draw "no face" warning but don't change alert state
            cv2.putText(
                frame, "No face detected",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 165, 255), 2, cv2.LINE_AA,
            )
            if self.metrics.calibrating:
                self._draw_calibration_overlay(frame, 0.0)
            return frame

        left_eye, right_eye = self.detector.get_eyes(landmarks)
        mouth               = self.detector.get_mouth(landmarks)

        # Update metrics
        metrics_data = self.metrics.update(left_eye, right_eye)
        ear          = metrics_data['ear']
        perclos      = metrics_data['perclos']
        blink_rate   = metrics_data['blink_rate']

        # MAR (always calculate, even during calibration — useful for display)
        mar = self.metrics.calculate_mar(mouth) if mouth is not None else 0.0


        # ── during calibration: show overlay only, no alerting ────────────
        if metrics_data['calibrating']:
            if self.show_landmarks:
                frame = self.detector.draw_landmarks(frame, landmarks)
            self._draw_calibration_overlay(frame, ear)
            return frame

        # ── normal detection ──────────────────────────────────────────────

        # Update consecutive counters
        if ear < self.metrics.ear_threshold:
            self.consecutive_drowsy += 1
        else:
            self.consecutive_drowsy = max(0, self.consecutive_drowsy - 1)

        if mar > self.mar_threshold:
            self.consecutive_yawn += 1
        else:
            self.consecutive_yawn = max(0, self.consecutive_yawn - 1)

        # Determine and trigger alert
        alert_level = self.determine_alert_level(perclos)
        if alert_level != "OK":
            self.alerts.trigger(alert_level, {'ear': ear, 'perclos': perclos or 0.0})

        # Log frame
        self.logger.log(
            frame=self.frame_count,
            ear=ear,
            mar=mar,
            perclos=perclos or 0.0,
            blink_rate=blink_rate or 0.0,
            alert_level=alert_level,
        )
        self.frame_count += 1

        # Draw landmarks and UI
        if self.show_landmarks:
            frame = self.detector.draw_landmarks(frame, landmarks)

        fps = self.camera.get_fps()
        frame = self.draw_ui(frame, ear, mar, perclos, blink_rate, alert_level, fps)

        return frame

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            self.camera.start()

            # Update metrics with the real FPS the camera is delivering
            # (wait two frames so get_fps() has data)
            import time
            time.sleep(0.1)
            real_fps = self.camera.get_fps()
            if real_fps > 5:
                self.metrics.fps          = real_fps
                self.metrics._cal_target  = int(
                    DrowsinessMetrics.CALIBRATION_SECONDS * real_fps
                )
                self.metrics.blink_history = __import__('collections').deque(
                    maxlen=int(real_fps * 5)
                )
                print(f"📷 Camera running at {real_fps:.1f} fps — calibration target: {self.metrics._cal_target} frames")

            window_name = self.config.get("display", {}).get("window_name", "Drowsiness Detection")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            

            print("\n🟡 Starting calibration — look straight ahead for 30 seconds.\n")

            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break

                frame = self.process_frame(frame)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Manual recalibrate
                    self.metrics.recalibrate()
                

        finally:
            self.cleanup()

    def cleanup(self):
        self.logger.close()
        self.camera.release()
        self.detector.cleanup()
        cv2.destroyAllWindows()
        print("\n✅ Session ended.")


if __name__ == "__main__":
    system = DrowsinessDetectionSystem()
    system.run()
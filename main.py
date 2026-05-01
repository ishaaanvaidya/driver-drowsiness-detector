"""Driver Drowsiness Detection System - Laptop Demo."""
import cv2
import yaml
from pathlib import Path

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DrowsinessMetrics
from src.alerts import AlertSystem
from src.logger import SessionLogger


class DrowsinessDetectionSystem:
    ALERT_ORDER = ["OK", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self):
        config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        cam_cfg = self.config.get("camera", {})
        det_cfg = self.config.get("detection", {})
        alert_cfg = self.config.get("alerts", {})
        display_cfg = self.config.get("display", {})
        drowsy_cfg = self.config.get("drowsiness", {})

        self.camera = Camera(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
        )

        self.detector = FaceDetector(
            min_detection_confidence=det_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=det_cfg.get("min_tracking_confidence", 0.5),
        )

        self.metrics = DrowsinessMetrics(
            fps=cam_cfg.get("fps", 30),
            perclos_window_seconds=drowsy_cfg.get("perclos_window_seconds", 5),
            microsleep_seconds=drowsy_cfg.get("microsleep_seconds", 1.25),
            calibration_ratio=drowsy_cfg.get("calibration_ratio", 0.80),
            fallback_threshold=drowsy_cfg.get("fallback_ear_threshold", 0.21),
            closed_eye_ratio=drowsy_cfg.get("closed_eye_ratio", 0.90),
        )

        self.alerts = AlertSystem(cooldown_seconds=alert_cfg.get("cooldown_seconds", 3))
        self.logger = SessionLogger()
        self.frame_count = 0

        self.consecutive_drowsy = 0
        self.consecutive_yawn = 0

        self.show_landmarks = display_cfg.get("show_landmarks", True)
        self.show_metrics = display_cfg.get("show_metrics", True)
        self.show_fps = display_cfg.get("show_fps", True)

        self.mar_threshold = drowsy_cfg.get("mar_threshold", 0.6)
        self.ear_consecutive_frames = drowsy_cfg.get("ear_consecutive_frames", 15)
        self.mar_consecutive_frames = drowsy_cfg.get("mar_consecutive_frames", 15)
        self.perclos_threshold = drowsy_cfg.get("perclos_threshold", 0.20)
        self.score_low = drowsy_cfg.get("score_low", 25)
        self.score_medium = drowsy_cfg.get("score_medium", 45)
        self.score_high = drowsy_cfg.get("score_high", 70)
        self.score_critical = drowsy_cfg.get("score_critical", 90)
        self.pitch_threshold = drowsy_cfg.get("pitch_threshold", 18.0)
        self.yaw_threshold = drowsy_cfg.get("yaw_threshold", 30.0)
        self.roll_threshold = drowsy_cfg.get("roll_threshold", 25.0)
        self.pose_score_max = drowsy_cfg.get("pose_score_max", 30)
        self.eye_unreliable_up_pitch = drowsy_cfg.get("eye_unreliable_up_pitch", 25.0)
        self.eye_unreliable_yaw = drowsy_cfg.get("eye_unreliable_yaw", 45.0)
        self.microsleep_high_seconds = drowsy_cfg.get("microsleep_high_seconds", 1.5)
        self.microsleep_critical_seconds = drowsy_cfg.get("microsleep_critical_seconds", 2.5)
        self.pose_medium_score = drowsy_cfg.get("pose_medium_score", 12)
        self.pose_high_score = drowsy_cfg.get("pose_high_score", 22)

        self.consecutive_pitch_down = 0
        self.consecutive_yaw_away = 0
        self.consecutive_roll_tilt = 0

        self.drowsiness_score = 0.0
        self.alert_level = "OK"
        self._pending_alert_level = "OK"
        self._pending_alert_frames = 0
        self.pose_score = 0.0
        self.no_face_frames = 0
        self._last_fps_sync_loop = 0
        self._last_pose = None
        self.pose_update_interval = self.config.get("detection", {}).get("pose_update_interval", 2)

    def _clamp(self, value, low=0.0, high=1.0):
        return max(low, min(high, value))

    def eyes_are_reliable(self, pose):
        """Return False when head angle makes EAR/PERCLOS/microsleep unreliable."""
        if pose is None:
            return True
        values = (pose.get("pitch"), pose.get("yaw"), pose.get("roll"))
        if any(value is None for value in values):
            return False
        if any(not (-180.0 <= float(value) <= 180.0) for value in values):
            return False
        # Looking sharply up/back makes eyelid geometry collapse in the webcam view.
        if pose["pitch"] < -self.eye_unreliable_up_pitch:
            return False
        # Large side turns distort both eye aspect ratio and PERCLOS.
        if abs(pose["yaw"]) > self.eye_unreliable_yaw:
            return False
        return True

    def _draw_calibration_overlay(self, frame, current_ear):
        h, w = frame.shape[:2]
        progress = self.metrics.calibration_progress
        seconds_done = int(DrowsinessMetrics.CALIBRATION_SECONDS * progress)
        seconds_left = DrowsinessMetrics.CALIBRATION_SECONDS - seconds_done

        frame[0:115, :] = (frame[0:115, :] * 0.5).astype(frame.dtype)

        cv2.putText(
            frame,
            "CALIBRATING - look straight ahead and keep eyes open",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 100),
            2,
            cv2.LINE_AA,
        )

        bar_x, bar_y, bar_h = 20, 50, 20
        bar_max_w = w - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)

        fill_w = int(bar_max_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 220, 130), -1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (120, 120, 120), 1)

        cv2.putText(
            frame,
            f"{int(progress * 100)}%",
            (bar_x + bar_max_w // 2 - 20, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        samples = len(self.metrics._cal_samples)
        cv2.putText(
            frame,
            f"{seconds_left}s remaining | samples: {samples} | EAR: {current_ear:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

    def calculate_pose_score(self, pose):
        """Convert sustained head movement into a capped soft risk score."""
        if pose is None:
            self.consecutive_pitch_down = 0
            self.consecutive_yaw_away = 0
            self.consecutive_roll_tilt = 0
            self.pose_score = max(0.0, self.pose_score - 0.5)
            return self.pose_score

        fps = self.camera.get_fps() or self.metrics.fps or 30

        if pose["pitch"] < -self.eye_unreliable_up_pitch:
            self.consecutive_pitch_down = 0

        # In this solvePnP setup, positive pitch means head down.
        # Head up is ignored because it is not a drowsiness signal.
        if pose["pitch"] > self.pitch_threshold:
            self.consecutive_pitch_down += 1
        else:
            self.consecutive_pitch_down = max(0, self.consecutive_pitch_down - 5)

        if abs(pose["yaw"]) > self.yaw_threshold:
            self.consecutive_yaw_away += 1
        else:
            self.consecutive_yaw_away = max(0, self.consecutive_yaw_away - 5)

        # Roll becomes unreliable when yaw is large, so freeze it during side turns.
        if abs(pose["yaw"]) < 20.0:
            if abs(pose["roll"]) > self.roll_threshold:
                self.consecutive_roll_tilt += 1
            else:
                self.consecutive_roll_tilt = max(0, self.consecutive_roll_tilt - 5)

        pitch_score = self._clamp((self.consecutive_pitch_down / fps - 0.8) / 1.7) * 24
        yaw_score = self._clamp((self.consecutive_yaw_away / fps - 2.0) / 2.5) * 5
        roll_score = self._clamp((self.consecutive_roll_tilt / fps - 1.5) / 2.0) * 7

        raw_pose_score = min(self.pose_score_max, pitch_score + yaw_score + roll_score)
        if raw_pose_score > self.pose_score:
            alpha = 0.25
        elif raw_pose_score == 0:
            alpha = 0.45
        else:
            alpha = 0.18
        self.pose_score = (1 - alpha) * self.pose_score + alpha * raw_pose_score
        return self.pose_score

    def calculate_fusion_score(self, ear, mar, perclos, blink_rate, microsleep_duration, pose_score, eyes_reliable=True):
        """Combine all available signals into one smoothed 0-100 score."""
        perclos = perclos or 0.0

        if eyes_reliable:
            ear_drop = max(0.0, self.metrics.closed_eye_threshold - ear)
            ear_score = self._clamp(ear_drop / max(self.metrics.closed_eye_threshold * 0.45, 1e-6)) * 18
            perclos_score = self._clamp((perclos - 0.10) / 0.30) * 30
            microsleep_score = self._clamp(microsleep_duration / max(self.microsleep_critical_seconds, 1e-6)) * 42
        else:
            ear_score = 0.0
            perclos_score = 0.0
            microsleep_score = 0.0

        yawn_score = self._clamp(self.consecutive_yawn / max(self.mar_consecutive_frames, 1)) * 8

        blink_score = 0.0
        if eyes_reliable and blink_rate is not None:
            if blink_rate < 8:
                blink_score = 6
            elif blink_rate > 45:
                blink_score = 4

        raw_score = min(
            100.0,
            ear_score + perclos_score + microsleep_score + yawn_score + blink_score + pose_score,
        )

        if raw_score > self.drowsiness_score:
            alpha = 0.35
        elif not eyes_reliable:
            alpha = 0.35
        else:
            alpha = 0.08
        self.drowsiness_score = (1 - alpha) * self.drowsiness_score + alpha * raw_score
        return self.drowsiness_score

    def _target_alert_from_score(self, score, microsleep_duration, eyes_reliable=True):
        if not eyes_reliable:
            return "OK"

        if self.pose_score >= self.pose_high_score:
            return "HIGH"
        if self.pose_score >= self.pose_medium_score:
            return "MEDIUM"

        if eyes_reliable and microsleep_duration >= self.microsleep_critical_seconds:
            return "CRITICAL"
        if eyes_reliable and microsleep_duration >= self.microsleep_high_seconds:
            return "HIGH"

        if score >= self.score_critical:
            return "CRITICAL"
        if score >= self.score_high:
            return "HIGH"
        if score >= self.score_medium:
            return "MEDIUM"
        if score >= self.score_low:
            return "LOW"
        return "OK"

    def update_alert_state(self, score, microsleep_duration, eyes_reliable=True):
        """State machine: prevents instant alert jumps from short noisy events."""
        if not eyes_reliable and self.pose_score < 2:
            self.alert_level = "OK"
            self._pending_alert_level = "OK"
            self._pending_alert_frames = 0
            return self.alert_level

        target = self._target_alert_from_score(score, microsleep_duration, eyes_reliable)

        if target == self.alert_level:
            self._pending_alert_level = target
            self._pending_alert_frames = 0
            return self.alert_level

        if target != self._pending_alert_level:
            self._pending_alert_level = target
            self._pending_alert_frames = 1
        else:
            self._pending_alert_frames += 1

        current_i = self.ALERT_ORDER.index(self.alert_level)
        target_i = self.ALERT_ORDER.index(target)
        fps = self.camera.get_fps() or self.metrics.fps or 30

        if target_i > current_i:
            seconds_needed = 0.6 if target == "CRITICAL" else 0.9 if target == "HIGH" else 0.8
        else:
            seconds_needed = 1.5

        if self._pending_alert_frames >= int(fps * seconds_needed):
            self.alert_level = target
            self._pending_alert_frames = 0

        return self.alert_level

    def draw_ui(self, frame, ear, mar, perclos, blink_rate, alert_level, fps, score, microsleep_duration, pose=None, pose_score=0.0, eyes_reliable=True):
        h, w = frame.shape[:2]
        header_h = 104
        frame[0:header_h, :] = (frame[0:header_h, :] * 0.45).astype(frame.dtype)

        if self.show_fps:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        if self.show_metrics:
            cv2.putText(
                frame,
                f"EAR: {ear:.3f} (thr: {self.metrics.ear_threshold:.3f})",
                (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            mar_color = (0, 0, 255) if mar > self.mar_threshold else (0, 255, 0)
            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                mar_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.1%}",
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            blink_text = "warming up" if blink_rate is None else f"{blink_rate:.1f}"
            cv2.putText(
                frame,
                f"Blinks/min: {blink_text}",
                (330, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            score_color = (0, 255, 0) if score < 25 else (0, 255, 255) if score < 65 else (0, 0, 255)
            cv2.putText(
                frame,
                f"Score: {score:.0f}/100  Microsleep: {microsleep_duration:.1f}s",
                (330, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                score_color,
                1,
                cv2.LINE_AA,
            )

            pose_text = "Pose: unavailable"
            if pose is not None:
                pose_text = (
                    f"Pose: P {pose['pitch']:+.1f}  "
                    f"Y {pose['yaw']:+.1f}  R {pose['roll']:+.1f}"
                )
            cv2.putText(
                frame,
                pose_text,
                (330, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        if alert_level and alert_level != "OK":
            colors = {
                "LOW": (0, 255, 255),
                "MEDIUM": (0, 165, 255),
                "HIGH": (0, 0, 255),
                "CRITICAL": (0, 0, 200),
            }
            color = colors.get(alert_level, (255, 255, 255))
            frame[h - 60:h, :] = (frame[h - 60:h, :] * 0.4).astype(frame.dtype)
            cv2.putText(
                frame,
                f"! {alert_level} DROWSINESS ALERT",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

        bar_max_w = 600
        bar_x = (w - bar_max_w) // 2
        bar_y = h - 80
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 12), (50, 50, 50), -1)
        fill_w = min(int((score / 100.0) * bar_max_w), bar_max_w)
        fill_color = (0, 200, 0) if score < 25 else (0, 180, 255) if score < 65 else (0, 0, 255)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), fill_color, -1)

        if self.metrics.baseline_ear is not None:
            cv2.putText(
                frame,
                f"baseline: {self.metrics.baseline_ear:.3f}",
                (w - 210, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        return frame

    def process_frame(self, frame):
        landmarks = self.detector.detect(frame)

        if landmarks is None:
            self.no_face_frames += 1
            self.metrics.reset_eye_state()
            self.drowsiness_score *= 0.85
            self.pose_score *= 0.85
            fps = self.camera.get_fps() or self.metrics.fps or 30
            if self.no_face_frames > int(fps):
                self.alert_level = "OK"
                self._pending_alert_level = "OK"
                self._pending_alert_frames = 0
            cv2.putText(
                frame,
                "No face detected",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            if self.metrics.calibrating:
                self._draw_calibration_overlay(frame, 0.0)
            return frame

        self.no_face_frames = 0

        left_eye, right_eye = self.detector.get_eyes(landmarks)
        mouth = self.detector.get_mouth(landmarks)
        h_frame, w_frame = frame.shape[:2]
        if self.frame_count % max(1, int(self.pose_update_interval)) == 0:
            self._last_pose = self.detector.get_head_pose(landmarks, w_frame, h_frame)
        pose = self._last_pose
        eyes_reliable = self.eyes_are_reliable(pose)

        metrics_data = self.metrics.update(left_eye, right_eye, eyes_reliable=eyes_reliable)
        ear = metrics_data["ear"]
        perclos = metrics_data["perclos"]
        blink_rate = metrics_data["blink_rate"]
        microsleep_duration = metrics_data["microsleep_duration"]

        mar = self.metrics.calculate_mar(mouth) if mouth is not None else 0.0

        if metrics_data["calibrating"]:
            if self.show_landmarks:
                frame = self.detector.draw_landmarks(frame, landmarks)
            self._draw_calibration_overlay(frame, ear)
            return frame

        if eyes_reliable and ear < self.metrics.closed_eye_threshold:
            self.consecutive_drowsy += 1
        else:
            self.consecutive_drowsy = 0

        if mar > self.mar_threshold:
            self.consecutive_yawn += 1
        else:
            self.consecutive_yawn = 0

        pose_score = self.calculate_pose_score(pose)
        score = self.calculate_fusion_score(
            ear,
            mar,
            perclos,
            blink_rate,
            microsleep_duration,
            pose_score,
            eyes_reliable=eyes_reliable,
        )
        alert_level = self.update_alert_state(score, microsleep_duration, eyes_reliable=eyes_reliable)

        if alert_level != "OK":
            self.alerts.trigger(
                alert_level,
                {"ear": ear, "perclos": perclos or 0.0, "score": score, "microsleep": microsleep_duration},
            )

        self.logger.log(
            frame=self.frame_count,
            ear=ear,
            mar=mar,
            perclos=perclos or 0.0,
            blink_rate=blink_rate,
            alert_level=alert_level,
            score=score,
            microsleep_duration=microsleep_duration,
            pose=pose,
            pose_score=pose_score,
            eyes_reliable=eyes_reliable,
        )
        self.frame_count += 1

        if self.show_landmarks:
            frame = self.detector.draw_landmarks(frame, landmarks)

        fps = self.camera.get_fps()
        return self.draw_ui(
            frame,
            ear,
            mar,
            perclos or 0.0,
            blink_rate,
            alert_level,
            fps,
            score,
            microsleep_duration,
            pose,
            pose_score,
            eyes_reliable,
        )

    def run(self):
        try:
            self.camera.start()

            import time

            time.sleep(0.1)
            real_fps = self.camera.get_reported_fps()
            if real_fps > 5:
                self.metrics.update_fps(real_fps)
                print(
                    f"Camera running at {real_fps:.1f} fps - "
                    f"calibration target: {self.metrics._cal_target} frames"
                )

            window_name = self.config.get("display", {}).get("window_name", "Drowsiness Detection")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                window_name,
                self.config.get("display", {}).get("window_width", 960),
                self.config.get("display", {}).get("window_height", 540),
            )

            print("\nStarting calibration - look straight ahead for 30 seconds.\n")

            loop_frames = 0
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    break

                loop_frames += 1
                if loop_frames - self._last_fps_sync_loop >= 120:
                    measured_fps = self.camera.get_fps()
                    if measured_fps > 5:
                        self.metrics.update_fps(measured_fps)
                        self._last_fps_sync_loop = loop_frames

                frame = self.process_frame(frame)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("l"):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks {'ON' if self.show_landmarks else 'OFF'}")
                if key == ord("r"):
                    self.metrics.recalibrate()
                    self.consecutive_drowsy = 0
                    self.consecutive_yawn = 0
                    self.consecutive_pitch_down = 0
                    self.consecutive_yaw_away = 0
                    self.consecutive_roll_tilt = 0
                    self.drowsiness_score = 0.0
                    self.pose_score = 0.0
                    self.alert_level = "OK"
                    self._pending_alert_level = "OK"
                    self._pending_alert_frames = 0
                    self._last_pose = None

        finally:
            self.cleanup()

    def cleanup(self):
        self.logger.close()
        self.camera.release()
        self.detector.cleanup()
        cv2.destroyAllWindows()
        print("\nSession ended.")


if __name__ == "__main__":
    system = DrowsinessDetectionSystem()
    system.run()

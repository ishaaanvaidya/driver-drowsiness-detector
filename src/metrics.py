"""Drowsiness metrics with better calibration, PERCLOS, and microsleep."""
from collections import deque

import numpy as np
from scipy.spatial import distance


class DrowsinessMetrics:
    """Calculate EAR, MAR, PERCLOS, blink rate, and microsleep duration."""

    CALIBRATION_SECONDS = 30
    CALIBRATION_RATIO = 0.80
    FALLBACK_THRESHOLD = 0.21

    # Internal "closed eye" threshold is stricter than display EAR threshold.
    # This prevents borderline open-eye EAR from being counted as closed.
    CLOSED_EYE_RATIO = 0.90
    MICROSLEEP_SECONDS = 1.25

    def __init__(
        self,
        fps=30,
        perclos_window_seconds=5,
        microsleep_seconds=None,
        calibration_ratio=None,
        fallback_threshold=None,
        closed_eye_ratio=None,
    ):
        self.fps = max(float(fps), 1.0)
        self.perclos_window_seconds = float(perclos_window_seconds)
        self.microsleep_seconds = float(microsleep_seconds if microsleep_seconds is not None else self.MICROSLEEP_SECONDS)
        self.calibration_ratio = float(calibration_ratio if calibration_ratio is not None else self.CALIBRATION_RATIO)
        self.fallback_threshold = float(fallback_threshold if fallback_threshold is not None else self.FALLBACK_THRESHOLD)
        self.closed_eye_ratio = float(closed_eye_ratio if closed_eye_ratio is not None else self.CLOSED_EYE_RATIO)

        self.ear_threshold = self.fallback_threshold
        self.closed_eye_threshold = self.ear_threshold * self.closed_eye_ratio

        self.ear_history = deque(maxlen=max(1, int(self.fps * self.perclos_window_seconds)))
        self.blink_history = deque(maxlen=max(1, int(self.fps * 60)))

        self.eye_was_open = True
        self._closed_frames = 0
        self._frame_index = 0
        self._blink_events = deque()

        self._min_blink_frames = max(2, int(self.fps * 0.08))
        self._closure_min_frames = max(6, int(self.fps * 0.25))
        self._microsleep_frames = max(1, int(self.fps * self.microsleep_seconds))

        self._cal_samples = []
        self._cal_target = int(self.CALIBRATION_SECONDS * self.fps)
        self.calibrating = True
        self.baseline_ear = None

    def update_fps(self, fps):
        """Update all FPS-dependent windows after the real camera FPS is known."""
        if fps <= 5:
            return

        self.fps = float(fps)

        old_ear = list(self.ear_history)
        ear_maxlen = max(1, int(self.fps * self.perclos_window_seconds))
        self.ear_history = deque(old_ear[-ear_maxlen:], maxlen=ear_maxlen)

        old_blink_history = list(self.blink_history)
        blink_maxlen = max(1, int(self.fps * 60))
        self.blink_history = deque(old_blink_history[-blink_maxlen:], maxlen=blink_maxlen)

        self._cal_target = int(self.CALIBRATION_SECONDS * self.fps)
        self._min_blink_frames = max(2, int(self.fps * 0.08))
        self._closure_min_frames = max(6, int(self.fps * 0.25))
        self._microsleep_frames = max(1, int(self.fps * self.microsleep_seconds))

    @property
    def calibration_progress(self):
        return min(len(self._cal_samples) / max(self._cal_target, 1), 1.0)

    def _finish_calibration(self):
        """Use robust open-eye samples instead of raw median of all samples."""
        samples = np.array(self._cal_samples, dtype=np.float64)
        samples = samples[samples > 0.05]

        if len(samples) < 10:
            self.baseline_ear = self.fallback_threshold
            self.ear_threshold = self.fallback_threshold
            print("\nCalibration had too few samples. Using fallback EAR threshold.")
        else:
            # Remove blink/lost-face lows and rare high spikes.
            low_cut = np.percentile(samples, 40)
            high_cut = np.percentile(samples, 98)
            open_eye_samples = samples[(samples >= low_cut) & (samples <= high_cut)]
            if len(open_eye_samples) < 10:
                open_eye_samples = samples

            self.baseline_ear = float(np.median(open_eye_samples))
            self.ear_threshold = round(self.baseline_ear * self.calibration_ratio, 4)
            self.closed_eye_threshold = round(self.ear_threshold * self.closed_eye_ratio, 4)

            print(
                f"\nCalibration complete - baseline EAR: {self.baseline_ear:.3f} | "
                f"threshold: {self.ear_threshold:.3f} | "
                f"closed threshold: {self.closed_eye_threshold:.3f}"
            )

        self.closed_eye_threshold = round(self.ear_threshold * self.closed_eye_ratio, 4)
        self.calibrating = False
        self.ear_history.clear()
        self.blink_history.clear()
        self._blink_events.clear()
        self._closed_frames = 0
        self._frame_index = 0
        self.eye_was_open = True

    def recalibrate(self):
        self._cal_samples = []
        self.calibrating = True
        self.baseline_ear = None
        self.ear_threshold = self.fallback_threshold
        self.closed_eye_threshold = self.fallback_threshold * self.closed_eye_ratio
        self.ear_history.clear()
        self.blink_history.clear()
        self._blink_events.clear()
        self._closed_frames = 0
        self._frame_index = 0
        self.eye_was_open = True
        print("\nRecalibrating - look straight ahead for 30 seconds.")

    def reset_eye_state(self):
        """Clear live closure state when the face/eyes are not reliable."""
        self._closed_frames = 0
        self.eye_was_open = True
        if self.blink_history:
            self.blink_history.append(True)

    def calculate_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth):
        vertical = distance.euclidean(mouth[0], mouth[1])
        horizontal = distance.euclidean(mouth[2], mouth[3])
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def calculate_perclos(self):
        """PERCLOS using stricter closed-eye threshold and sustained runs."""
        if not self.ear_history:
            return 0.0

        closed = 0
        closed_run = 0

        for ear in self.ear_history:
            if ear < self.closed_eye_threshold:
                closed_run += 1
            else:
                if closed_run >= self._closure_min_frames:
                    closed += closed_run
                closed_run = 0

        if closed_run >= self._closure_min_frames:
            closed += closed_run

        return closed / len(self.ear_history)

    def _update_blink_and_closure(self, ear):
        eye_is_closed = ear < self.closed_eye_threshold

        if eye_is_closed:
            self._closed_frames += 1
        else:
            if self._min_blink_frames <= self._closed_frames < self._microsleep_frames:
                self._blink_events.append(self._frame_index)
            self._closed_frames = 0

        self.eye_was_open = not eye_is_closed
        self.blink_history.append(self.eye_was_open)

    def get_blink_rate(self):
        """Return blinks/min after a short warmup; None means not ready."""
        min_frames = int(self.fps * 20)
        if self._frame_index < min_frames:
            return None

        window_frames = int(self.fps * 60)
        cutoff = self._frame_index - window_frames
        while self._blink_events and self._blink_events[0] < cutoff:
            self._blink_events.popleft()

        duration_s = min(self._frame_index, window_frames) / self.fps
        if duration_s <= 0:
            return None
        return (len(self._blink_events) / duration_s) * 60

    def get_microsleep_duration(self):
        return self._closed_frames / self.fps

    def update(self, left_eye, right_eye, eyes_reliable=True):
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if self.calibrating:
            if eyes_reliable and avg_ear > 0.05:
                self._cal_samples.append(avg_ear)

            if len(self._cal_samples) >= self._cal_target:
                self._finish_calibration()

            return {
                "ear": avg_ear,
                "perclos": None,
                "blink_rate": None,
                "microsleep_duration": 0.0,
                "microsleeping": False,
                "calibrating": True,
                "eyes_reliable": eyes_reliable,
            }

        self._frame_index += 1

        if not eyes_reliable:
            # At extreme head angles, EAR is geometrically unreliable. Treat the
            # frame as "not closed" so head-up/side-angle views do not create
            # false microsleep or PERCLOS spikes.
            safe_ear = max(avg_ear, self.ear_threshold)
            self.ear_history.append(safe_ear)
            self.reset_eye_state()

            return {
                "ear": avg_ear,
                "perclos": self.calculate_perclos(),
                "blink_rate": self.get_blink_rate(),
                "microsleep_duration": 0.0,
                "microsleeping": False,
                "calibrating": False,
                "eyes_reliable": False,
            }

        self.ear_history.append(avg_ear)
        self._update_blink_and_closure(avg_ear)

        microsleep_duration = self.get_microsleep_duration()

        return {
            "ear": avg_ear,
            "perclos": self.calculate_perclos(),
            "blink_rate": self.get_blink_rate(),
            "microsleep_duration": microsleep_duration,
            "microsleeping": microsleep_duration >= self.microsleep_seconds,
            "calibrating": False,
            "eyes_reliable": True,
        }

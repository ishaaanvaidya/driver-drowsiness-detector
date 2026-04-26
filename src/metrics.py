"""Drowsiness metrics calculation — with per-session EAR calibration"""
import numpy as np
from scipy.spatial import distance
from collections import deque


class DrowsinessMetrics:
    """Calculate EAR, MAR, PERCLOS, blink rate — threshold personalised per session"""

    # ── calibration constants ─────────────────────────────────────────────────
    CALIBRATION_SECONDS = 30      # how long to collect baseline samples
    CALIBRATION_RATIO   = 0.85    # ear_threshold = baseline × this
    FALLBACK_THRESHOLD  = 0.21    # used if no face detected during calibration

    def __init__(self, fps=30, perclos_window_seconds=3):
        self.fps = fps

        # Threshold starts at fallback; overwritten when calibration finishes
        self.ear_threshold = self.FALLBACK_THRESHOLD

        # PERCLOS rolling window
        perclos_frames = int(fps * perclos_window_seconds)
        self.ear_history = deque(maxlen=perclos_frames)

        # Blink history — 5 s worth of frames
        self.blink_history = deque(maxlen=int(fps * 5))
        self.eye_was_open  = True

        # ── calibration state ─────────────────────────────────────────────
        self._cal_samples = []                              # raw EAR values collected
        self._cal_target  = int(self.CALIBRATION_SECONDS * fps)  # frames needed
        self.calibrating  = True                            # flips False when done
        self.baseline_ear = None                            # set on completion

    # ── calibration ───────────────────────────────────────────────────────────

    @property
    def calibration_progress(self):
        """Float 0.0–1.0 showing how far through the 30-second window we are."""
        return min(len(self._cal_samples) / max(self._cal_target, 1), 1.0)

    def _finish_calibration(self):
        """Lock threshold from collected samples. Falls back if samples are sparse."""
        if len(self._cal_samples) < 10:
            # Face was barely detected — warn and use fallback
            self.baseline_ear  = self.FALLBACK_THRESHOLD
            self.ear_threshold = self.FALLBACK_THRESHOLD
            print(
                "\n⚠️  Calibration: too few samples — "
                f"falling back to threshold {self.FALLBACK_THRESHOLD}. "
                "Make sure your face is visible and try again."
            )
        else:
            # Median ignores blink outliers and brief face-loss frames
            self.baseline_ear  = float(np.median(self._cal_samples))
            self.ear_threshold = round(self.baseline_ear * self.CALIBRATION_RATIO, 4)
            print(
                f"\n✅ Calibration complete — "
                f"baseline EAR: {self.baseline_ear:.3f}  |  "
                f"threshold set to: {self.ear_threshold:.3f}  "
                f"({len(self._cal_samples)} samples)"
            )

        self.calibrating = False

    def recalibrate(self):
        """Reset calibration so the next session re-runs the 30-second window."""
        self._cal_samples  = []
        self.calibrating   = True
        self.baseline_ear  = None
        self.ear_threshold = self.FALLBACK_THRESHOLD
        self.ear_history.clear()
        self.blink_history.clear()
        self.eye_was_open  = True
        print("\n🔄 Recalibrating — look straight ahead for 30 seconds.")

    # ── metric formulas ───────────────────────────────────────────────────────

    def calculate_ear(self, eye):
        """
        Eye Aspect Ratio.
        eye: (6, 2) array — [outer, top1, top2, inner, bottom2, bottom1]
        Returns 0.0 on degenerate geometry.
        """
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth):
        """
        Mouth Aspect Ratio.
        mouth: (4, 2) array — [top, bottom, left, right]
        Returns 0.0 on degenerate geometry.
        """
        vertical   = distance.euclidean(mouth[0], mouth[1])
        horizontal = distance.euclidean(mouth[2], mouth[3])
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def calculate_perclos(self):
        """
        Percentage of eyelid closure over the rolling window.
        Returns 0.0 until at least 10 frames are collected.
        """
        if len(self.ear_history) < 10:
            return 0.0
        closed = sum(1 for e in self.ear_history if e < self.ear_threshold)
        return closed / len(self.ear_history)

    def update_blink(self, ear):
        """Record open/closed state; detect open→closed transitions."""
        eye_is_open = ear >= self.ear_threshold
        self.eye_was_open = eye_is_open
        self.blink_history.append(eye_is_open)

    def get_blink_rate(self):
        """
        Blinks per minute using actual fps.
        Normal: 15–20 bpm. Drowsy: <10 bpm.
        """
        if len(self.blink_history) < 30:
            return 0.0
        transitions = sum(
            1 for i in range(1, len(self.blink_history))
            if self.blink_history[i - 1] and not self.blink_history[i]
        )
        duration_s = len(self.blink_history) / self.fps
        if duration_s == 0:
            return 0.0
        return (transitions / duration_s) * 60

    # ── main update — called every frame ─────────────────────────────────────

    def update(self, left_eye, right_eye):
        """
        Update all metrics.

        During calibration (first 30 s), collects samples and returns
        perclos=None, blink_rate=None so the caller knows not to alert.

        Returns:
            dict with keys: ear, perclos, blink_rate, calibrating
        """
        left_ear  = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── calibration phase ─────────────────────────────────────────────
        if self.calibrating:
            # Only record frames where a real face is visible (EAR > 0)
            if avg_ear > 0.05:
                self._cal_samples.append(avg_ear)

            if len(self._cal_samples) >= self._cal_target:
                self._finish_calibration()

            return {
                'ear':         avg_ear,
                'perclos':     None,
                'blink_rate':  None,
                'calibrating': True,
            }

        # ── normal detection phase ────────────────────────────────────────
        self.ear_history.append(avg_ear)
        self.update_blink(avg_ear)

        return {
            'ear':         avg_ear,
            'perclos':     self.calculate_perclos(),
            'blink_rate':  self.get_blink_rate(),
            'calibrating': False,
        }
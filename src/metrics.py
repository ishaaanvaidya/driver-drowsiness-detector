"""Drowsiness metrics calculation"""
import numpy as np
from scipy.spatial import distance
from collections import deque


class DrowsinessMetrics:
    """Calculate EAR, MAR, PERCLOS, blink rate"""
    
    def __init__(self, ear_threshold=0.21, perclos_window_frames=90):
        self.ear_threshold = ear_threshold
        self.perclos_window = perclos_window_frames
        
        # History buffers
        self.ear_history = deque(maxlen=perclos_window_frames)
        self.blink_history = deque(maxlen=150)  # 5 seconds @ 30fps
        
        # State
        self.eye_was_open = True
        self.blink_count = 0
        
    def calculate_ear(self, eye):
        """
        Eye Aspect Ratio
        
        Args:
            eye: (6, 2) array of eye landmarks
                [outer, top1, top2, inner, bottom2, bottom1]
        
        Returns:
            EAR value (0.2-0.4, <0.25 = closed)
        """
        # Vertical distances
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Horizontal distance
        C = distance.euclidean(eye[0], eye[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth):
        """
        Mouth Aspect Ratio
        
        Args:
            mouth: (4, 2) array [top, bottom, left, right]
        
        Returns:
            MAR value (>0.6 = yawning)
        """
        # Vertical distance
        vertical = distance.euclidean(mouth[0], mouth[1])
        
        # Horizontal distance
        horizontal = distance.euclidean(mouth[2], mouth[3])
        
        # MAR formula
        mar = vertical / horizontal
        return mar
    
    def calculate_perclos(self):
        """
        Percentage of Eyelid Closure
        
        Returns:
            PERCLOS value (>0.15 = drowsy)
        """
        if len(self.ear_history) < 10:
            return 0.0
        
        closed_count = sum(1 for ear in self.ear_history if ear < self.ear_threshold)
        perclos = closed_count / len(self.ear_history)
        
        return perclos
    
    def update_blink(self, ear):
        """Update blink detection"""
        eye_is_open = ear >= self.ear_threshold
        
        # Detect blink (open -> closed transition)
        if self.eye_was_open and not eye_is_open:
            self.blink_count += 1
        
        self.eye_was_open = eye_is_open
        self.blink_history.append(eye_is_open)
    
    def get_blink_rate(self):
        """
        Blinks per minute
        
        Returns:
            Blink rate (normal: 15-20, drowsy: <10)
        """
        if len(self.blink_history) < 30:
            return 0.0
        
        # Count transitions
        transitions = 0
        for i in range(1, len(self.blink_history)):
            if self.blink_history[i-1] and not self.blink_history[i]:
                transitions += 1
        
        # Convert to blinks per minute
        duration_seconds = len(self.blink_history) / 30.0
        bpm = (transitions / duration_seconds) * 60
        
        return bpm
    
    def update(self, left_eye, right_eye):
        """
        Update all metrics
        
        Returns:
            dict with ear, perclos, blink_rate
        """
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Update history
        self.ear_history.append(avg_ear)
        self.update_blink(avg_ear)
        
        return {
            'ear': avg_ear,
            'perclos': self.calculate_perclos(),
            'blink_rate': self.get_blink_rate()
        }
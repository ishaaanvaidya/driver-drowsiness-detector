"""Simple camera handler"""
import cv2
import time
import numpy as np


class Camera:
    """Webcam capture with FPS tracking"""
    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        # FPS tracking
        self.frame_times = []
        self.last_time = time.time()
        
    def start(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")
        
        print(f"✅ Camera started: {self.width}x{self.height} @ {self.fps}fps")
        
    def read(self):
        """Read frame and calculate FPS"""
        ret, frame = self.cap.read()
        
        if ret:
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time)
            self.last_time = current_time
            
            self.frame_times.append(fps)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
        
        return ret, frame
    
    def get_fps(self):
        """Get average FPS"""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times)
    
    def release(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
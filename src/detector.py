"""MediaPipe face detection"""
import cv2
import numpy as np
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW = True
except ImportError:
    import mediapipe as mp
    MEDIAPIPE_NEW = False



class FaceDetector:
    """MediaPipe Face Mesh detector"""
    
    # Eye landmark indices (MediaPipe 478-point model)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    # Mouth landmark indices
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Try to import the correct mediapipe module
        try:
            from mediapipe.python.solutions import face_mesh as fm
            self.face_mesh = fm.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except ImportError:
            # Fallback for older versions
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
    def detect(self, frame):
        """
        Detect face landmarks
        
        Returns:
            landmarks: (478, 2) array of pixel coordinates or None
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face = results.multi_face_landmarks[0]
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        landmarks = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in face.landmark
        ])
        
        return landmarks
    
    def get_eyes(self, landmarks):
        """Extract eye landmarks"""
        if landmarks is None:
            return None, None
        
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        return left_eye, right_eye
    
    def get_mouth(self, landmarks):
        """Extract mouth landmarks"""
        if landmarks is None:
            return None
        
        mouth = np.array([
            landmarks[self.MOUTH_TOP],
            landmarks[self.MOUTH_BOTTOM],
            landmarks[self.MOUTH_LEFT],
            landmarks[self.MOUTH_RIGHT],
        ])
        
        return mouth
    
    def draw_landmarks(self, frame, landmarks):
        """Draw eye and mouth landmarks"""
        if landmarks is None:
            return frame
        
        result = frame.copy()
        
        # Draw eyes
        for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = landmarks[eye_indices]
            for i in range(len(eye_points)):
                p1 = tuple(eye_points[i])
                p2 = tuple(eye_points[(i + 1) % len(eye_points)])
                cv2.line(result, p1, p2, (0, 255, 0), 2)
        
        # Draw mouth
        mouth = self.get_mouth(landmarks)
        if mouth is not None:
            cv2.line(result, tuple(mouth[0]), tuple(mouth[1]), (0, 0, 255), 2)
            cv2.line(result, tuple(mouth[2]), tuple(mouth[3]), (0, 0, 255), 2)
        
        return result
    
    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()
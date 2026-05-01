"""MediaPipe face detection with EAR/MAR landmarks and head pose."""
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class FaceDetector:
    """MediaPipe Face Mesh detector optimized for real-time drowsiness detection."""

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308

    POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]
    FACE_3D_MODEL = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ], dtype=np.float64)
    DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if mp is None:
            raise ImportError("mediapipe is required for FaceDetector")

        try:
            from mediapipe.python.solutions import face_mesh as fm
            face_mesh = fm
        except ImportError:
            face_mesh = mp.solutions.face_mesh

        self.face_mesh = face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._camera_matrix = None
        self._frame_size = None

    def _get_camera_matrix(self, frame_w, frame_h):
        """Build a simple webcam camera matrix for solvePnP."""
        if self._frame_size == (frame_w, frame_h):
            return self._camera_matrix

        focal_length = frame_w
        cx = frame_w / 2
        cy = frame_h / 2

        self._camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self._frame_size = (frame_w, frame_h)
        return self._camera_matrix

    def detect(self, frame):
        """Return face landmarks as pixel coordinates, or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        return np.array([
            [lm.x * w, lm.y * h]
            for lm in face.landmark
        ], dtype=np.float64)

    def get_eyes(self, landmarks):
        """Extract left and right eye landmarks."""
        if landmarks is None:
            return None, None
        return landmarks[self.LEFT_EYE], landmarks[self.RIGHT_EYE]

    def get_mouth(self, landmarks):
        """Extract mouth landmarks as [top, bottom, left, right]."""
        if landmarks is None:
            return None
        return np.array([
            landmarks[self.MOUTH_TOP],
            landmarks[self.MOUTH_BOTTOM],
            landmarks[self.MOUTH_LEFT],
            landmarks[self.MOUTH_RIGHT],
        ])

    def get_head_pose(self, landmarks, frame_w, frame_h):
        """Estimate pitch, yaw, and roll in degrees using cv2.solvePnP."""
        if landmarks is None or len(landmarks) <= max(self.POSE_LANDMARK_IDS):
            return None

        image_points = np.array([landmarks[i] for i in self.POSE_LANDMARK_IDS], dtype=np.float64)
        camera_matrix = self._get_camera_matrix(frame_w, frame_h)

        success, rvec, _ = cv2.solvePnP(
            self.FACE_3D_MODEL,
            image_points,
            camera_matrix,
            self.DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    def draw_landmarks(self, frame, landmarks):
        """Draw anti-aliased eye and mouth guides directly on the frame."""
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]
        thickness = 2 if min(h, w) >= 540 else 1

        for eye_indices in (self.LEFT_EYE, self.RIGHT_EYE):
            pts = landmarks[eye_indices].round().astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=thickness, lineType=cv2.LINE_AA)

        mouth = self.get_mouth(landmarks)
        if mouth is not None:
            mouth = mouth.round().astype(np.int32)
            cv2.line(frame, tuple(mouth[0]), tuple(mouth[1]), (0, 0, 255), thickness, cv2.LINE_AA)
            cv2.line(frame, tuple(mouth[2]), tuple(mouth[3]), (0, 0, 255), thickness, cv2.LINE_AA)

        return frame

    def cleanup(self):
        self.face_mesh.close()

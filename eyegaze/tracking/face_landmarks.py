from __future__ import annotations

from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceLandmarksResult:
    landmarks: np.ndarray
    image_width: int
    image_height: int


class FaceLandmarkTracker:
    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr) -> FaceLandmarksResult | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        face = result.multi_face_landmarks[0]
        pts = []
        for lm in face.landmark:
            pts.append([lm.x * w, lm.y * h, lm.z])
        return FaceLandmarksResult(np.asarray(pts, dtype=np.float32), w, h)

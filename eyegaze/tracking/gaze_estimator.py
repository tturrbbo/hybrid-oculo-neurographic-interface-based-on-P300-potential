from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from eyegaze.tracking.face_landmarks import FaceLandmarkTracker
from eyegaze.tracking.face_geometry_head import FaceGeometryHeadEstimator
from eyegaze.tracking.distance import DistanceEstimator
from eyegaze.tracking.head_deviation import get_head_deviation

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_VERTICAL = [159, 145]
RIGHT_EYE_VERTICAL = [386, 374]
NOSE_TIP = 1


@dataclass
class GazeResult:
    features: np.ndarray | None
    blink: bool
    gaze_xy: tuple[int, int] | None
    meta: dict


class GazeEstimator:
    """
    Надёжные углы головы:
    - НЕ используем solvePnP для отображаемых углов.
    - Используем стабильный proxy по носу и глазам.
    - Потом переводим proxy в реальные градусы через предварительную
      калибровку по меткам 1-10.
    """

    def __init__(self, baseline_distance_cm: float = 60.0, tolerance_cm: float = 7.0, head_angle_cfg: dict | None = None):
        head_angle_cfg = head_angle_cfg or {}
        self.head_angle_mode = str(head_angle_cfg.get("mode", "face_geometry")).lower()
        self.face_geometry = FaceGeometryHeadEstimator(head_angle_cfg.get("face_geometry", {}))
        self.landmarks = FaceLandmarkTracker()
        self.distance = DistanceEstimator(baseline_distance_cm, tolerance_cm)

        self.model = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=220,
                random_state=42,
                min_samples_leaf=1,
                max_features=None,
            )
        )
        self.knn_model = KNeighborsRegressor(n_neighbors=3, weights="distance")

        self.left_eye_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=120, random_state=43, min_samples_leaf=1)
        )
        self.right_eye_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=120, random_state=44, min_samples_leaf=1)
        )

        self.is_trained = False
        self.has_knn = False
        self.has_eye_models = False

        # Постобработка откалиброванного gaze:
        # deadzone убирает микродрожание, KNN помогает держаться ближе
        # к реально собранным калибровочным точкам.
        self.knn_weight = 0.50
        self.eye_model_weight = 0.10
        self.gaze_deadzone_px = 14.0
        self.gaze_ema_alpha = 0.55
        self._last_gaze_xy_float = None

        self.dominant_eye = "BOTH"
        self.left_eye_calibration_error_px = None
        self.right_eye_calibration_error_px = None
        self.left_eye_weight = 0.5
        self.right_eye_weight = 0.5

        self.baseline_yaw_deg = None
        self.baseline_pitch_deg = None

        # Для marker-based calibration знаки задаются самими метками.
        self.invert_yaw = False
        self.invert_pitch = False

        self.head_angle_calibration = None

    def _center(self, lm, idx):
        return lm[idx, :2].mean(axis=0)

    def _blink(self, lm):
        lh = np.linalg.norm(lm[LEFT_EYE_CORNERS[0], :2] - lm[LEFT_EYE_CORNERS[1], :2])
        lv = np.linalg.norm(lm[LEFT_EYE_VERTICAL[0], :2] - lm[LEFT_EYE_VERTICAL[1], :2])
        rh = np.linalg.norm(lm[RIGHT_EYE_CORNERS[0], :2] - lm[RIGHT_EYE_CORNERS[1], :2])
        rv = np.linalg.norm(lm[RIGHT_EYE_VERTICAL[0], :2] - lm[RIGHT_EYE_VERTICAL[1], :2])
        ratio = (lv / max(lh, 1e-6) + rv / max(rh, 1e-6)) / 2.0
        return bool(ratio < 0.12)

    def _head_proxy_from_landmarks(self, lm):
        if self.head_angle_mode == "face_geometry":
            g = self.face_geometry.estimate(lm)
            if g.valid:
                return g.yaw_deg, g.pitch_deg
            return None, None

        try:
            left_eye = self._center(lm, LEFT_EYE_CORNERS)
            right_eye = self._center(lm, RIGHT_EYE_CORNERS)
            eye_mid = (left_eye + right_eye) / 2.0
            eye_dist = float(np.linalg.norm(left_eye - right_eye))

            if eye_dist < 1e-6:
                return None, None

            nose = lm[NOSE_TIP, :2]

            # Это НЕ градусы, а стабильная шкала положения головы.
            # Реальные градусы получаются после калибровки по меткам.
            yaw_proxy = float((nose[0] - eye_mid[0]) / eye_dist * 60.0)
            pitch_proxy = float((nose[1] - eye_mid[1]) / eye_dist * 60.0)

            return yaw_proxy, pitch_proxy
        except Exception:
            return None, None

    def calibrate_distance_baseline_from_frame(self, frame_bgr):
        res = self.landmarks.process(frame_bgr)
        if res is None:
            return None
        px = self.distance.calibrate_baseline(res.landmarks)
        if self.head_angle_mode == "face_geometry":
            self.face_geometry.calibrate_neutral(res.landmarks)
            # Keep old baseline fields valid for existing UI/logger code.
            self.baseline_yaw_deg = 0.0
            self.baseline_pitch_deg = 0.0
        return px

    def get_head_pose_from_frame(self, frame_bgr):
        res = self.landmarks.process(frame_bgr)
        if res is None:
            return None, None
        if self.head_angle_mode == "face_geometry" and not self.face_geometry.is_calibrated():
            self.face_geometry.calibrate_neutral(res.landmarks)
            self.baseline_yaw_deg = 0.0
            self.baseline_pitch_deg = 0.0
        return self._head_proxy_from_landmarks(res.landmarks)

    def set_head_pose_baseline(self, yaw_values, pitch_values):
        yaw_values = [float(v) for v in yaw_values if v is not None]
        pitch_values = [float(v) for v in pitch_values if v is not None]

        if len(yaw_values) < 1 or len(pitch_values) < 1:
            return False

        self.baseline_yaw_deg = float(np.median(yaw_values))
        self.baseline_pitch_deg = float(np.median(pitch_values))

        print(f"[head] proxy baseline_yaw={self.baseline_yaw_deg:.3f}")
        print(f"[head] proxy baseline_pitch={self.baseline_pitch_deg:.3f}")
        return True

    def set_head_angle_calibration_from_marker_samples(self, samples):
        if self.baseline_yaw_deg is None or self.baseline_pitch_deg is None:
            return False

        horizontal_points = [(0.0, 0.0)]
        vertical_points = [(0.0, 0.0)]

        for s in samples:
            direction = str(s.get("direction", "")).upper()
            angle = float(s.get("angle_deg", 0.0))
            yaw = s.get("yaw_deg")
            pitch = s.get("pitch_deg")

            if yaw is None or pitch is None:
                continue

            raw_yaw = float(yaw) - float(self.baseline_yaw_deg)
            raw_pitch = float(pitch) - float(self.baseline_pitch_deg)

            if self.invert_yaw:
                raw_yaw = -raw_yaw
            if self.invert_pitch:
                raw_pitch = -raw_pitch

            if direction == "RIGHT":
                horizontal_points.append((raw_yaw, +angle))
            elif direction == "LEFT":
                horizontal_points.append((raw_yaw, -angle))
            elif direction == "UP":
                vertical_points.append((raw_pitch, +angle))
            elif direction == "DOWN":
                vertical_points.append((raw_pitch, -angle))

        def clean(points):
            out = []
            for x, y in sorted(points, key=lambda p: p[0]):
                if out and abs(out[-1][0] - x) < 1e-4:
                    out[-1] = ((out[-1][0] + x) / 2.0, (out[-1][1] + y) / 2.0)
                else:
                    out.append((float(x), float(y)))
            return out

        self.head_angle_calibration = {
            "horizontal_points": clean(horizontal_points),
            "vertical_points": clean(vertical_points),
            "samples": samples,
        }

        print("[head] reliable marker-based angle calibration complete")
        print(f"[head] horizontal proxy->deg: {self.head_angle_calibration['horizontal_points']}")
        print(f"[head] vertical proxy->deg: {self.head_angle_calibration['vertical_points']}")
        return True

    @staticmethod
    def _left_features(features):
        return np.asarray(
            [features[0], features[1], features[4], features[5], features[6], features[7]],
            dtype=np.float32,
        )

    @staticmethod
    def _right_features(features):
        return np.asarray(
            [features[2], features[3], features[4], features[5], features[6], features[7]],
            dtype=np.float32,
        )

    def extract_features_meta(self, frame_bgr):
        res = self.landmarks.process(frame_bgr)
        if res is None:
            return None, True, {"face_found": False}

        lm = res.landmarks
        blink = self._blink(lm)

        left_iris = self._center(lm, LEFT_IRIS)
        right_iris = self._center(lm, RIGHT_IRIS)

        left_outer = lm[LEFT_EYE_CORNERS[0], :2]
        left_inner = lm[LEFT_EYE_CORNERS[1], :2]
        right_inner = lm[RIGHT_EYE_CORNERS[0], :2]
        right_outer = lm[RIGHT_EYE_CORNERS[1], :2]

        left_center = (left_outer + left_inner) / 2.0
        right_center = (right_outer + right_inner) / 2.0
        left_w = np.linalg.norm(left_outer - left_inner)
        right_w = np.linalg.norm(right_outer - right_inner)

        left_norm = (left_iris - left_center) / max(left_w, 1e-6)
        right_norm = (right_iris - right_center) / max(right_w, 1e-6)

        yaw_proxy, pitch_proxy = self._head_proxy_from_landmarks(lm)
        face_geometry_result = None
        if self.head_angle_mode == "face_geometry":
            face_geometry_result = self.face_geometry.estimate(lm)
            if face_geometry_result.valid:
                yaw_proxy = face_geometry_result.yaw_deg
                pitch_proxy = face_geometry_result.pitch_deg

        try:
            pose = self.head_pose.estimate(lm, res.image_width, res.image_height)
            roll = pose.get("roll_deg")
        except Exception:
            roll = None

        distance_cm, eye_px, dist_status = self.distance.estimate(lm)
        if face_geometry_result is not None and face_geometry_result.valid:
            distance_cm = face_geometry_result.distance_cm or distance_cm
            eye_px = face_geometry_result.eye_distance_px or eye_px

        head_dev = get_head_deviation(
            yaw_proxy,
            pitch_proxy,
            baseline_yaw_deg=self.baseline_yaw_deg,
            baseline_pitch_deg=self.baseline_pitch_deg,
            invert_yaw=self.invert_yaw,
            invert_pitch=self.invert_pitch,
            head_angle_calibration=self.head_angle_calibration,
        )

        lx, ly = float(left_norm[0]), float(left_norm[1])
        rx, ry = float(right_norm[0]), float(right_norm[1])

        yn = 0.0 if yaw_proxy is None else float(yaw_proxy) / 45.0
        pn = 0.0 if pitch_proxy is None else float(pitch_proxy) / 45.0
        rn = 0.0 if roll is None else float(roll) / 45.0
        dn = 0.0 if distance_cm is None else float(distance_cm) / 60.0

        avg_x = (lx + rx) / 2.0
        avg_y = (ly + ry) / 2.0
        diff_x = lx - rx
        diff_y = ly - ry

        features = np.array(
            [
                lx, ly, rx, ry,
                yn, pn, rn, dn,
                lx * ly, rx * ry,
                lx * lx, ly * ly, rx * rx, ry * ry,
                avg_y, avg_x,
                diff_x, diff_y,
                avg_x * avg_y,
                avg_y * avg_y,
                pn * avg_y,
                yn * avg_x,
            ],
            dtype=np.float32,
        )

        meta = {
            "face_found": True,
            "yaw_deg": yaw_proxy,
            "pitch_deg": pitch_proxy,
            "roll_deg": roll,
            "baseline_yaw_deg": self.baseline_yaw_deg,
            "baseline_pitch_deg": self.baseline_pitch_deg,
            "distance_cm": distance_cm,
            "eye_distance_px": eye_px,
            "distance_status": dist_status,
            "head_angle_mode": self.head_angle_mode,
            "face_geometry_valid": None if face_geometry_result is None else bool(face_geometry_result.valid),
            "face_geometry_reason": "" if face_geometry_result is None else face_geometry_result.reason,
            "brow_distance_px": None if face_geometry_result is None else face_geometry_result.brow_distance_px,
            "blink": blink,
            "dominant_eye": self.dominant_eye,
            "left_eye_calibration_error_px": self.left_eye_calibration_error_px,
            "right_eye_calibration_error_px": self.right_eye_calibration_error_px,
            "left_eye_weight": self.left_eye_weight,
            "right_eye_weight": self.right_eye_weight,
        }
        meta.update(head_dev)
        return features, blink, meta

    @staticmethod
    def _mean_error_px(pred, target):
        diff = pred - target
        return float(np.mean(np.linalg.norm(diff, axis=1)))

    def _choose_dominant_eye(self, left_err, right_err):
        self.left_eye_calibration_error_px = float(left_err)
        self.right_eye_calibration_error_px = float(right_err)

        if abs(left_err - right_err) < 8.0:
            self.dominant_eye = "BOTH"
            self.left_eye_weight = 0.5
            self.right_eye_weight = 0.5
            return

        if left_err < right_err:
            self.dominant_eye = "LEFT"
            self.left_eye_weight = 0.65
            self.right_eye_weight = 0.35
        else:
            self.dominant_eye = "RIGHT"
            self.left_eye_weight = 0.35
            self.right_eye_weight = 0.65

    def train(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.model.fit(X, y)
        self.is_trained = True

        if len(X) >= 8:
            k = min(3, len(X))
            self.knn_model = KNeighborsRegressor(n_neighbors=k, weights="distance")
            self.knn_model.fit(X, y)
            self.has_knn = True

        if len(X) >= 20:
            X_left = np.asarray([self._left_features(row) for row in X], dtype=np.float32)
            X_right = np.asarray([self._right_features(row) for row in X], dtype=np.float32)

            self.left_eye_model.fit(X_left, y)
            self.right_eye_model.fit(X_right, y)
            self.has_eye_models = True

            left_pred = self.left_eye_model.predict(X_left)
            right_pred = self.right_eye_model.predict(X_right)

            self._choose_dominant_eye(
                self._mean_error_px(left_pred, y),
                self._mean_error_px(right_pred, y),
            )
        else:
            self.has_eye_models = False
            self.dominant_eye = "BOTH"
            self.left_eye_weight = 0.5
            self.right_eye_weight = 0.5

        print("[gaze] Calibration complete")
        print(f"[gaze] dominant_eye={self.dominant_eye}")

    def set_gaze_postprocess_config(self, cfg: dict | None):
        """
        Настройки устойчивости gaze после калибровки.

        Пример в experiment.yaml:
        gaze_tiles:
          knn_weight: 0.55
          eye_model_weight: 0.10
          gaze_deadzone_px: 14
          gaze_ema_alpha: 0.55
        """
        cfg = cfg or {}
        self.knn_weight = float(cfg.get("knn_weight", self.knn_weight))
        self.eye_model_weight = float(cfg.get("eye_model_weight", self.eye_model_weight))
        self.gaze_deadzone_px = float(cfg.get("gaze_deadzone_px", self.gaze_deadzone_px))
        self.gaze_ema_alpha = float(cfg.get("gaze_ema_alpha", self.gaze_ema_alpha))

        self.knn_weight = max(0.0, min(1.0, self.knn_weight))
        self.eye_model_weight = max(0.0, min(1.0, self.eye_model_weight))
        self.gaze_ema_alpha = max(0.0, min(1.0, self.gaze_ema_alpha))
        self.gaze_deadzone_px = max(0.0, self.gaze_deadzone_px)

    def _postprocess_gaze_xy(self, pred):
        pred = np.asarray(pred, dtype=np.float32)

        if self._last_gaze_xy_float is None:
            self._last_gaze_xy_float = pred
            return float(pred[0]), float(pred[1])

        last = np.asarray(self._last_gaze_xy_float, dtype=np.float32)
        dist = float(np.linalg.norm(pred - last))

        # Если сдвиг совсем маленький — считаем это шумом landmarks.
        if dist < float(self.gaze_deadzone_px):
            pred2 = last
        else:
            a = float(self.gaze_ema_alpha)
            pred2 = a * pred + (1.0 - a) * last

        self._last_gaze_xy_float = pred2
        return float(pred2[0]), float(pred2[1])

    def predict_xy(self, features):
        if not self.is_trained:
            raise RuntimeError("Gaze model is not trained. Run calibration first.")

        features = np.asarray(features, dtype=np.float32)
        common_xy = self.model.predict(np.asarray([features], dtype=np.float32))[0]

        if self.has_knn:
            knn_xy = self.knn_model.predict(np.asarray([features], dtype=np.float32))[0]
            kw = float(self.knn_weight)
            pred = common_xy * (1.0 - kw) + knn_xy * kw
        else:
            pred = common_xy

        if self.has_eye_models:
            left_f = self._left_features(features)
            right_f = self._right_features(features)

            left_xy = self.left_eye_model.predict(np.asarray([left_f], dtype=np.float32))[0]
            right_xy = self.right_eye_model.predict(np.asarray([right_f], dtype=np.float32))[0]

            eye_xy = left_xy * self.left_eye_weight + right_xy * self.right_eye_weight
            ew = float(self.eye_model_weight)
            pred = pred * (1.0 - ew) + eye_xy * ew

        return self._postprocess_gaze_xy(pred)

    def process_frame(self, frame_bgr):
        features, blink, meta = self.extract_features_meta(frame_bgr)
        gaze_xy = None

        if features is not None and not blink and self.is_trained:
            gaze_xy = self.predict_xy(features)

        meta["dominant_eye"] = self.dominant_eye
        meta["left_eye_calibration_error_px"] = self.left_eye_calibration_error_px
        meta["right_eye_calibration_error_px"] = self.right_eye_calibration_error_px
        meta["left_eye_weight"] = self.left_eye_weight
        meta["right_eye_weight"] = self.right_eye_weight

        return GazeResult(features, blink, gaze_xy, meta)

    def save_model(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)

    def load_model(self, path: str | Path):
        data = joblib.load(path)
        self.__dict__.update(data)

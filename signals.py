"""GuardWatch per-frame / per-track signal helpers.

Keeps heavy math out of guardwatch.py. Stateless functions are plain; any
state (PERCLOS rolling buffer, calibrator) is encapsulated in small classes.
"""


def oklid_mesafe(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def ear_hesapla(e1, e2):
    p11, p12, p13, p14, p15, p16 = e1
    p21, p22, p23, p24, p25, p26 = e2
    e1_ear = (oklid_mesafe(p12, p16) + oklid_mesafe(p13, p15)) / (2 * oklid_mesafe(p11, p14))
    e2_ear = (oklid_mesafe(p22, p26) + oklid_mesafe(p23, p25)) / (2 * oklid_mesafe(p21, p24))
    return (e1_ear + e2_ear) / 2.0


import cv2


def upscale_roi(roi, min_size):
    """Scale a BGR image so its longest side is at least `min_size` pixels.

    Returns the input unchanged when:
      - roi is None
      - max(h, w) < 20 (too small; MediaPipe would fail regardless)
      - max(h, w) >= min_size (already large enough)

    Uses cv2.INTER_CUBIC and preserves aspect ratio.
    """
    if roi is None:
        return None
    h, w = roi.shape[:2]
    longest = max(h, w)
    if longest < 20 or longest >= min_size:
        return roi
    scale = min_size / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


from collections import deque


class PerclosBuffer:
    """Per-track rolling buffer of (timestamp, is_closed) tuples.

    percent(now) prunes entries older than `window_sec` and returns the
    percentage of remaining entries that were `is_closed=True`.
    """

    def __init__(self, window_sec):
        self.window_sec = float(window_sec)
        self._buf = deque()

    def add(self, now, is_closed):
        self._buf.append((float(now), bool(is_closed)))

    def _prune(self, now):
        cutoff = now - self.window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def percent(self, now):
        self._prune(float(now))
        if not self._buf:
            return 0.0
        closed = sum(1 for _, c in self._buf if c)
        return 100.0 * closed / len(self._buf)


import numpy as np
import math


# Indices into MediaPipe FaceLandmarker's 478-point face mesh
_HEAD_POSE_LANDMARK_IDS = (
    1,    # nose tip
    152,  # chin
    33,   # left eye outer corner
    263,  # right eye outer corner
    61,   # left mouth corner
    291,  # right mouth corner
)

# 3D model points in millimetres, matching the order above. Uses OpenCV
# image convention (Y down) so a frontal face → identity rotation and
# pitch ≈ 0°. Offsets calibrated to typical face geometry.
_MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),       # nose tip
    (0.0,    63.6,  -12.5),      # chin (below nose)
    (-43.3, -10.6,  -26.0),      # left eye outer (slightly above nose)
    (43.3,  -10.6,  -26.0),      # right eye outer
    (-28.9,  31.9,  -24.1),      # left mouth corner (below nose)
    (28.9,   31.9,  -24.1),      # right mouth corner
], dtype=np.float64)


def compute_head_pose(face_landmarks, roi_shape):
    """Return (pitch, roll, yaw) in degrees from MediaPipe face landmarks.

    face_landmarks: sequence of objects with `.x` and `.y` attributes in [0,1]
    roi_shape: (h, w) of the image the landmarks are expressed in

    Returns (0.0, 0.0, 0.0) on any failure (missing landmarks, solvePnP failure).
    """
    try:
        import cv2
        h, w = roi_shape
        image_points = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in _HEAD_POSE_LANDMARK_IDS],
            dtype=np.float64,
        )
        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal, 0,     center[0]],
            [0,     focal, center[1]],
            [0,     0,     1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rvec)
        # Extract Euler angles (ZYX convention) from the rotation matrix.
        sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll  = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
            yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll  = 0.0
        # Fold any leftover half-turn from OpenCV version drift so a
        # frontal face reads ~0°.
        if pitch > 90.0:
            pitch -= 180.0
        elif pitch < -90.0:
            pitch += 180.0
        return float(pitch), float(roll), float(yaw)
    except Exception:
        return 0.0, 0.0, 0.0


import logging as _logging
import time as _time


class GlobalCalibrator:
    """Learn a shared EAR baseline from the first N seconds of usage.

    Samples fed via update() are collected until `calibration_duration_sec`
    passes. On the first ear_closed() call after that window, the threshold
    is computed as `percentile * ratio` and cached. Before that, the
    fallback `ear_threshold` from config is returned.
    """

    def __init__(self, config, now_fn=None):
        self._duration    = float(config["calibration_duration_sec"])
        self._min_samples = int(config["calibration_min_samples"])
        self._percentile  = float(config["calibration_ear_percentile"])
        self._ratio       = float(config["ear_closed_ratio"])
        self._fallback    = float(config["ear_threshold"])
        self._now_fn      = now_fn if now_fn is not None else _time.time
        self._start       = self._now_fn()
        self._samples     = []
        self._cached      = None
        self._finalized   = False

    def update(self, ear_value):
        # Only accept samples during the calibration window.
        if self._finalized:
            return
        if (self._now_fn() - self._start) < self._duration:
            self._samples.append(float(ear_value))

    def ear_closed(self):
        if self._finalized:
            return self._cached
        if (self._now_fn() - self._start) < self._duration:
            return self._fallback
        # Window just closed — finalize exactly once.
        self._finalized = True
        if len(self._samples) < self._min_samples:
            _logging.warning(
                f"Kalibrasyon yetersiz ornek ({len(self._samples)} < {self._min_samples}); "
                f"varsayilan esige donuluyor ({self._fallback})"
            )
            self._cached = self._fallback
            return self._cached
        import numpy as _np
        baseline = float(_np.percentile(self._samples, self._percentile))
        self._cached = baseline * self._ratio
        _logging.info(
            f"Kalibrasyon tamamlandi: baseline={baseline:.3f}, threshold={self._cached:.3f} "
            f"(n={len(self._samples)} ornek)"
        )
        return self._cached

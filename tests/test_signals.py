import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals import ear_hesapla


def test_ear_hesapla_open_eye_returns_high_ratio():
    # 6 points forming an "open eye" shape:
    # p1,p4 are horizontal corners (wide); p2,p3,p5,p6 are vertical mid-points (tall)
    eye = [(0, 5), (3, 0), (7, 0), (10, 5), (7, 10), (3, 10)]
    result = ear_hesapla(eye, eye)
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) = (10+10) / (2*10) = 1.0
    assert abs(result - 1.0) < 1e-6


def test_ear_hesapla_closed_eye_returns_low_ratio():
    # Same horizontal width, but vertical distances near zero
    eye = [(0, 5), (3, 5), (7, 5), (10, 5), (7, 5), (3, 5)]
    result = ear_hesapla(eye, eye)
    # Vertical distances are zero → EAR ≈ 0
    assert result < 0.01


import numpy as np

from signals import upscale_roi


def test_upscale_roi_small_is_upscaled_longest_side_384():
    roi = np.zeros((60, 80, 3), dtype=np.uint8)  # h=60, w=80 — longest side 80
    out = upscale_roi(roi, min_size=384)
    h, w = out.shape[:2]
    assert max(h, w) == 384
    # aspect ratio preserved: original 60/80 = 0.75, new h/w == 0.75
    assert abs((h / w) - (60 / 80)) < 0.02


def test_upscale_roi_large_is_unchanged():
    roi = np.zeros((500, 500, 3), dtype=np.uint8)
    out = upscale_roi(roi, min_size=384)
    assert out.shape == roi.shape
    # also: should return the exact same array (no copy needed)
    assert out is roi


def test_upscale_roi_tiny_returned_unchanged():
    roi = np.zeros((10, 15, 3), dtype=np.uint8)  # below 20 px floor
    out = upscale_roi(roi, min_size=384)
    assert out is roi


def test_upscale_roi_none_returned_unchanged():
    assert upscale_roi(None, min_size=384) is None


from signals import PerclosBuffer


def test_perclos_empty_buffer_returns_zero():
    buf = PerclosBuffer(window_sec=60)
    assert buf.percent(now=1000.0) == 0.0


def test_perclos_all_closed_returns_100():
    buf = PerclosBuffer(window_sec=60)
    for t in range(10):
        buf.add(float(t), True)
    assert buf.percent(now=10.0) == 100.0


def test_perclos_half_closed_returns_50():
    buf = PerclosBuffer(window_sec=60)
    for t in range(10):
        buf.add(float(t), t % 2 == 0)
    assert buf.percent(now=10.0) == 50.0


def test_perclos_prunes_old_entries():
    buf = PerclosBuffer(window_sec=60)
    # 10 closed entries at t=0..9 (old)
    for t in range(10):
        buf.add(float(t), True)
    # 10 open entries at t=100..109 (current)
    for t in range(100, 110):
        buf.add(float(t), False)
    # At now=110 with window=60, old entries (t<50) are pruned
    assert buf.percent(now=110.0) == 0.0


from signals import compute_head_pose


class _FakeLm:
    """Stand-in for a MediaPipe NormalizedLandmark — only .x and .y are used."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _build_landmark_list(points):
    """points is {index: (x, y)} in normalized [0,1]. Returns a list where
    unused indices are filled with dummy (0, 0) landmarks."""
    max_idx = max(points)
    lst = [_FakeLm(0.0, 0.0) for _ in range(max_idx + 1)]
    for i, (x, y) in points.items():
        lst[i] = _FakeLm(x, y)
    return lst


def test_compute_head_pose_frontal_face_returns_small_angles():
    # Build a symmetric frontal face: nose centered, eyes/mouth symmetric around x=0.5
    points = {
        1:   (0.50, 0.50),   # nose tip
        152: (0.50, 0.80),   # chin
        33:  (0.35, 0.45),   # left eye outer
        263: (0.65, 0.45),   # right eye outer
        61:  (0.40, 0.65),   # left mouth corner
        291: (0.60, 0.65),   # right mouth corner
    }
    lms = _build_landmark_list(points)
    pitch, roll, yaw = compute_head_pose(lms, (480, 640))
    assert abs(pitch) < 20.0
    assert abs(roll) < 10.0
    assert abs(yaw) < 20.0


def test_compute_head_pose_failure_returns_zeros():
    # Empty landmark list — cannot index into required points, must not raise.
    pitch, roll, yaw = compute_head_pose([], (480, 640))
    assert (pitch, roll, yaw) == (0.0, 0.0, 0.0)


from signals import GlobalCalibrator


def _cal_config(**overrides):
    base = {
        "calibration_duration_sec": 30,
        "calibration_min_samples": 5,
        "calibration_ear_percentile": 10,
        "ear_closed_ratio": 0.72,
        "ear_threshold": 0.21,
    }
    base.update(overrides)
    return base


def test_calibrator_pre_window_returns_fallback():
    t = [0.0]
    cal = GlobalCalibrator(_cal_config(), now_fn=lambda: t[0])
    cal.update(0.30)
    assert cal.ear_closed() == 0.21


def test_calibrator_post_window_with_samples_returns_calibrated():
    t = [0.0]
    cal = GlobalCalibrator(_cal_config(), now_fn=lambda: t[0])
    for v in [0.28, 0.30, 0.31, 0.29, 0.32, 0.27, 0.33]:
        cal.update(v)
    t[0] = 31.0
    threshold = cal.ear_closed()
    # 10th percentile of the samples above is ~0.276, times 0.72 ~= 0.199
    assert 0.18 < threshold < 0.22
    # Second call should return the cached value, not recompute.
    assert cal.ear_closed() == threshold


def test_calibrator_post_window_without_samples_returns_fallback():
    t = [0.0]
    cal = GlobalCalibrator(_cal_config(), now_fn=lambda: t[0])
    t[0] = 31.0
    assert cal.ear_closed() == 0.21

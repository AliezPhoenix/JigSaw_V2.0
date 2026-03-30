# pytest tests/test_mark_detector_multi_roi.py
import numpy as np
import pytest

from src.detectors.mark_detector import MarkDetector


def _white_square_image(size=120):
    img = np.zeros((size, size), dtype=np.uint8)
    img[40:80, 40:80] = 255
    return img


@pytest.mark.parametrize(
    "allow_mark,rois,expect_valid",
    [
        (False, [[30, 30, 60, 60]], True),
        (True, [[30, 30, 60, 60]], True),
        (False, [[0, 0, 10, 10], [30, 30, 60, 60]], True),
        (True, [[0, 0, 10, 10], [30, 30, 60, 60]], False),
        (True, [[30, 30, 60, 60], [35, 35, 50, 50]], True),
    ],
)
def test_mark_detector_aggregation(allow_mark, rois, expect_valid):
    img = _white_square_image()
    n = len(rois)
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "mark_detect_mode": "manual",
            "pixel_size": 0.001,
            "mark_rois": rois,
            "mark_roi_min_areas": [50] * n,
            "allow_mark": allow_mark,
        }
    )
    ok, _msg, rd = det.detect(img)
    assert ok is True
    assert rd["is_valid"] is expect_valid
    assert "per_roi" in rd and len(rd["per_roi"]) == len(rois)


def test_empty_mark_rois_not_valid():
    img = _white_square_image()
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "mark_rois": [],
            "mark_roi_min_areas": [],
            "allow_mark": False,
        }
    )
    ok, _msg, rd = det.detect(img)
    assert ok is True
    assert rd["is_valid"] is False
    assert rd["per_roi"] == []


def test_invalid_min_area_negative_returns_failure():
    """mark_roi_min_areas 与 mark_rois 等长后若存在负数，detect 应失败。"""
    img = _white_square_image()
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "mark_rois": [[30, 30, 60, 60], [0, 0, 10, 10]],
            "mark_roi_min_areas": [50, -1],
            "allow_mark": False,
        }
    )
    ok, msg, rd = det.detect(img)
    assert ok is False
    assert "无效" in msg or "最小面积" in msg
    assert rd["per_roi"] == []


def test_per_roi_different_min_areas():
    """同一图像下两 ROI 使用不同 min_area：仅大阈值 ROI 能检出白块。"""
    img = _white_square_image()
    rois = [[30, 30, 60, 60], [0, 0, 20, 20]]
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "mark_detect_mode": "manual",
            "pixel_size": 0.001,
            "mark_rois": rois,
            "mark_roi_min_areas": [50, 50000],
            "allow_mark": False,
        }
    )
    ok, _msg, rd = det.detect(img)
    assert ok is True
    per = rd["per_roi"]
    assert per[0]["is_valid"] is True
    assert per[1]["is_valid"] is False

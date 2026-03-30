# pytest tests/test_mark_detector_multi_roi.py
import numpy as np
import pytest

from src.detectors.mark_detector import MarkDetector
from src.support.support_funs import normalize_mark_rois_in_params


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
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "min_mark_area": 50,
            "mark_detect_mode": "manual",
            "pixel_size": 0.001,
            "mark_rois": rois,
            "allow_mark": allow_mark,
        }
    )
    ok, _msg, rd = det.detect(img)
    assert ok is True
    assert rd["is_valid"] is expect_valid
    assert "per_roi" in rd and len(rd["per_roi"]) == len(rois)


def test_normalize_mark_roi_legacy():
    p = {"mark_roi": [1, 2, 30, 40], "foo": 1}
    normalize_mark_rois_in_params(p)
    assert p["mark_rois"] == [[1, 2, 30, 40]]
    assert "mark_roi" not in p


def test_empty_mark_rois_not_valid():
    img = _white_square_image()
    det = MarkDetector(
        {
            "min_threshold": 200,
            "max_threshold": 255,
            "min_mark_area": 50,
            "mark_rois": [],
            "allow_mark": False,
        }
    )
    ok, _msg, rd = det.detect(img)
    assert ok is True
    assert rd["is_valid"] is False
    assert rd["per_roi"] == []

"""Station work-image source: prefer BGA full frame, else current_image."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.support.support_funs import resolve_station_work_image, should_cache_detect_display_as_current


def test_prefer_selected_frame_over_current_image():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    current = np.ones((2, 2, 3), dtype=np.uint8)
    assert resolve_station_work_image(frame, current) is frame


def test_fallback_to_current_image_when_no_frame():
    current = np.ones((2, 2, 3), dtype=np.uint8)
    assert resolve_station_work_image(None, current) is current


def test_none_when_neither_source_exists():
    assert resolve_station_work_image(None, None) is None


def test_cache_detect_result_not_live_preview():
    assert should_cache_detect_display_as_current(object()) is True
    assert should_cache_detect_display_as_current(None) is False

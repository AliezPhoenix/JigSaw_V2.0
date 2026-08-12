"""DryPramasSetDialog auto pixel-size correction helpers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DryPramasSetDialog import DryPramasSetDialog


def test_calc_pixel_sizes_from_actual():
    ps_x, ps_y = DryPramasSetDialog._calc_pixel_sizes_from_actual(12.4, 15.0, 1240.0, 1500.0)
    assert ps_x == pytest.approx(0.01)
    assert ps_y == pytest.approx(0.01)

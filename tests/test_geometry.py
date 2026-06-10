import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from renderer.renderer import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
)


def test_center_basic():
    bbox = [100, 200, 300, 400]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 200
    assert cy == 300


def test_center_square():
    bbox = [0, 0, 100, 100]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 50
    assert cy == 50


def test_center_non_square():
    bbox = [10, 20, 110, 60]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 60
    assert cy == 40


def test_center_returns_integers():
    bbox = [0, 0, 101, 101]
    cx, cy = get_center_of_bbox(bbox)
    assert isinstance(cx, int)
    assert isinstance(cy, int)


def test_center_float_bbox():
    bbox = [10.5, 20.5, 50.5, 80.5]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 30
    assert cy == 50


def test_width_basic():
    bbox = [100, 0, 300, 0]
    assert get_bbox_width(bbox) == 200


def test_width_zero():
    bbox = [100, 0, 100, 0]
    assert get_bbox_width(bbox) == 0


def test_width_float():
    bbox = [10.5, 0, 50.5, 0]
    assert get_bbox_width(bbox) == pytest.approx(40.0)


def test_foot_is_bottom_center():
    bbox = [100, 200, 300, 400]
    fx, fy = get_foot_position(bbox)
    assert fx == 200
    assert fy == 400


def test_foot_y_equals_y2():
    bbox = [0, 50, 200, 350]
    _, fy = get_foot_position(bbox)
    assert fy == 350


def test_foot_returns_integers():
    bbox = [0.0, 0.0, 101.0, 201.0]
    fx, fy = get_foot_position(bbox)
    assert isinstance(fx, int)
    assert isinstance(fy, int)

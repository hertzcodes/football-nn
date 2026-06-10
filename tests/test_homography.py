import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from homography.homography import HomographyTransformer, PITCH_KEYPOINT_COORDS


class MockPitchTracker:
    def __init__(self, keypoints=None, confidences=None):
        self.keypoints = keypoints
        self.confidences = confidences

    def detect_keypoints_with_confidence(self, frame):
        if self.keypoints is None:
            return None
        return self.keypoints, self.confidences


def test_homography_init():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    assert ht.field_detector == tracker
    assert ht.H is None
    assert ht.keyframe_H == {}
    assert ht.sorted_keyframes == []


def test_interpolate_H():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    H1 = np.eye(3)
    H2 = np.eye(3) * 2.0
    H_interp = ht._interpolate_H(H1, H2, 0.5)
    expected = np.eye(3) * 1.5
    assert np.allclose(H_interp, expected)


def test_sanity_check():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    assert ht._sanity_check(np.eye(3))
    assert not ht._sanity_check(np.zeros((3, 3)))
    
    huge = np.eye(3)
    huge[0, 0] = 100.0
    huge[1, 1] = 100.0
    assert not ht._sanity_check(huge)


def test_transform():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    assert ht.transform((10, 20)) is None
    
    ht.H = np.eye(3)
    assert ht.transform((10, 20)) == (10, 20)
    
    H_trans = np.eye(3)
    H_trans[0, 2] = 5.0
    H_trans[1, 2] = 10.0
    ht.H = H_trans
    assert ht.transform((10, 20)) == (15, 30)


def test_compute_homography_valid():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    src_pts = PITCH_KEYPOINT_COORDS.copy()
    confidences = np.ones(len(src_pts))
    
    H = ht.compute(src_pts, confidences)
    assert H is not None
    assert np.allclose(H, np.eye(3), atol=1e-3)


def test_compute_homography_too_few_keypoints():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    src_pts = np.array([[0, 0], [100, 100], [200, 200]])
    confidences = np.ones(3)
    
    H = ht.compute(src_pts, confidences)
    assert H is None


def test_update_for_frame():
    tracker = MockPitchTracker()
    ht = HomographyTransformer(tracker)
    H0 = np.eye(3)
    H90 = np.eye(3) * 2.0
    ht.keyframe_H = {0: H0, 90: H90}
    ht.sorted_keyframes = [0, 90]
    
    ht.update_for_frame(0, None, np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.allclose(ht.H, H0)
    
    ht.update_for_frame(45, None, np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.allclose(ht.H, np.eye(3) * 1.5)
    
    ht.update_for_frame(90, None, np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.allclose(ht.H, H90)

    ht.update_for_frame(120, None, np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.allclose(ht.H, H90)

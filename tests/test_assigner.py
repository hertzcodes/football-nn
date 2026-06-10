import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from assigner.assigner import Assigner


def make_frame_with_colors(colors, bbox_size=60):
    n = len(colors)
    frame = np.zeros((200, n * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        x1 = i * 100 + 20
        y1 = 20
        x2 = x1 + bbox_size
        y2 = y1 + bbox_size
        frame[y1:y2, x1:x2] = color
    return frame


def make_bboxes(n, bbox_size=60):
    bboxes = []
    for i in range(n):
        x1 = i * 100 + 20
        y1 = 20
        x2 = x1 + bbox_size
        y2 = y1 + bbox_size
        bboxes.append([x1, y1, x2, y2])
    return bboxes


def test_assigner_init():
    a = Assigner()
    assert not a.fitted
    assert len(a.team1_ids) == 0
    assert len(a.team2_ids) == 0
    assert len(a.memory) == 0


def test_get_team_unknown_returns_0():
    a = Assigner()
    assert a.get_team(999) == 0


def test_get_team_never_returns_none():
    a = Assigner()
    for pid in range(100):
        result = a.get_team(pid)
        assert result is not None
        assert result in (0, 1, 2)


def test_assign_team_splits_two_colors():
    red = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, red, red, blue, blue, blue]
    frame = make_frame_with_colors(colors)
    bboxes = make_bboxes(len(colors))
    players = {i: {"bounding_box": bboxes[i]} for i in range(len(colors))}

    a = Assigner()
    a.assign_team(frame, players)

    assert a.fitted
    assert len(a.team1_ids) > 0
    assert len(a.team2_ids) > 0
    assert len(a.team1_ids) + len(a.team2_ids) == len(colors)


def test_assign_team_too_few_players():
    frame = make_frame_with_colors([(0, 0, 200)])
    players = {0: {"bounding_box": make_bboxes(1)[0]}}

    a = Assigner()
    a.assign_team(frame, players)

    assert not a.fitted


def test_assign_team_uses_real_track_ids():
    red = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, blue]
    frame = make_frame_with_colors(colors)
    bboxes = make_bboxes(2)
    players = {
        42: {"bounding_box": bboxes[0]},
        87: {"bounding_box": bboxes[1]},
    }

    a = Assigner()
    a.assign_team(frame, players)

    if a.fitted:
        all_ids = a.team1_ids | a.team2_ids
        assert 42 in all_ids or 87 in all_ids
        assert 0 not in all_ids
        assert 1 not in all_ids


def test_get_player_team_returns_track_id():
    red = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox = make_bboxes(1)[0]

    a = Assigner()
    pid = a.get_player_team(frame, bbox, track_id=5)
    assert pid == 5


def test_get_player_team_empty_crop_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [50, 50, 50, 50]

    a = Assigner()
    result = a.get_player_team(frame, bbox, track_id=1)
    assert result is None


def test_get_player_team_stores_in_memory():
    red = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox = make_bboxes(1)[0]

    a = Assigner()
    a.get_player_team(frame, bbox, track_id=7)
    assert 7 in a.memory


def test_get_player_team_same_id_twice():
    red = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox = make_bboxes(1)[0]

    a = Assigner()
    pid1 = a.get_player_team(frame, bbox, track_id=3)
    pid2 = a.get_player_team(frame, bbox, track_id=3)
    assert pid1 == pid2 == 3


def test_team_assignment_consistent():
    red = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox = make_bboxes(1)[0]

    a = Assigner()
    a.get_player_team(frame, bbox, track_id=10)
    team1 = a.get_team(10)
    team2 = a.get_team(10)
    assert team1 == team2


def test_no_player_in_both_teams():
    red = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, red, blue, blue]
    frame = make_frame_with_colors(colors)
    bboxes = make_bboxes(4)
    players = {i: {"bounding_box": bboxes[i]} for i in range(4)}

    a = Assigner()
    a.assign_team(frame, players)

    overlap = a.team1_ids & a.team2_ids
    assert len(overlap) == 0


def test_get_team_color():
    a = Assigner()
    assert a.get_team_color(0) == (0, 215, 255)
    assert a.get_team_color(999) == (0, 215, 255)


def test_get_team_color_after_assign():
    red = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, red, red, blue, blue, blue]
    frame = make_frame_with_colors(colors)
    bboxes = make_bboxes(len(colors))
    players = {i: {"bounding_box": bboxes[i]} for i in range(len(colors))}

    a = Assigner()
    a.assign_team(frame, players)

    color1 = a.get_team_color(1)
    color2 = a.get_team_color(2)
    assert color1 != (0, 215, 255)
    assert color2 != (0, 215, 255)
    assert len(color1) == 3
    assert len(color2) == 3


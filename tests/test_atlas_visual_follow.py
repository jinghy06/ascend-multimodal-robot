import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "esp32_voice_terminal"))

import atlas_visual_follow as avf  # noqa: E402


def test_person_velocity_left_and_far():
    cfg = avf.FollowConfig()
    # Person on left (cx=120 < 320) and far (height=144 -> h_ratio=0.30 < 0.58)
    box = avf.PersonBox(x1=80, y1=100, x2=160, y2=244, score=0.9)
    vx, omega = avf.compute_person_velocity(box, cfg)
    assert omega > 0, "Target on left should rotate CCW (positive omega)"
    assert vx > 0, "Far target should move forward (positive vx)"
    assert vx <= cfg.max_linear_fwd
    assert omega <= cfg.max_angular


def test_person_velocity_right_and_too_close():
    cfg = avf.FollowConfig()
    # Person on right (cx=500 > 320) and too close (height=384 -> h_ratio=0.80 > 0.58)
    box = avf.PersonBox(x1=420, y1=40, x2=580, y2=424, score=0.95)
    vx, omega = avf.compute_person_velocity(box, cfg)
    assert omega < 0, "Target on right should rotate CW (negative omega)"
    assert vx < 0, "Too-close target should gently reverse"
    assert vx >= -cfg.max_linear_rev


def test_person_velocity_deadband_zero():
    cfg = avf.FollowConfig()
    # Centered (cx=320), exact target height (0.58 * 480 = 278.4)
    box = avf.PersonBox(x1=240, y1=100, x2=400, y2=378.4, score=0.9)
    vx, omega = avf.compute_person_velocity(box, cfg)
    assert vx == 0.0
    assert omega == 0.0


def test_target_lock_keeps_previous_person():
    p_left = avf.PersonBox(x1=100, y1=100, x2=200, y2=340, score=0.9)
    p_right_big = avf.PersonBox(x1=420, y1=50, x2=580, y2=400, score=0.92)
    # Without history, picks larger box
    first = avf.select_person_target([p_left, p_right_big], None, 640, 480)
    assert first is p_right_big
    # With history on p_left, keeps tracking p_left even if p_right_big appears
    p_left_moved = avf.PersonBox(x1=110, y1=102, x2=210, y2=342, score=0.88)
    locked = avf.select_person_target([p_left_moved, p_right_big], p_left, 640, 480)
    assert locked is p_left_moved


def test_voice_state_machine_transitions():
    ctrl = avf.FollowController()
    assert ctrl.mode == "idle"

    ctrl.apply_voice_command({"task_id": "follow_start", "params": {}})
    assert ctrl.mode == "person"

    old_ratio = ctrl.cfg.target_h_ratio
    ctrl.apply_voice_command({"task_id": "follow_adjust", "params": {"distance": "near"}})
    assert ctrl.cfg.target_h_ratio > old_ratio
    ctrl.apply_voice_command({"task_id": "follow_adjust", "params": {"distance": "far"}})
    assert abs(ctrl.cfg.target_h_ratio - old_ratio) < 1e-6

    ctrl.apply_voice_command({"task_id": "pause", "params": {}})
    assert ctrl.mode == "idle"
    ctrl.apply_voice_command({"task_id": "resume", "params": {}})
    assert ctrl.mode == "person"

    ctrl.apply_voice_command({"task_id": "laser_start", "params": {}})
    assert ctrl.mode == "laser"
    ctrl.apply_voice_command({"task_id": "stop_all", "params": {}})
    assert ctrl.mode == "idle"
    # After stop_all, resume should stay idle
    ctrl.apply_voice_command({"task_id": "resume", "params": {}})
    assert ctrl.mode == "idle"


def test_target_lost_coast_search_stop():
    ctrl = avf.FollowController(initial_mode="person")
    # See person on the left at t=10.0
    box = avf.PersonBox(x1=80, y1=120, x2=180, y2=320, score=0.9)
    vx, omega, locked, alert = ctrl.step_person([box], now=10.0)
    assert locked is box and omega > 0 and alert is None

    # Coast phase (0.2s later)
    vx1, om1, locked1, alert1 = ctrl.step_person([], now=10.2)
    assert locked1 is None and alert1 is None
    assert abs(vx1) < abs(vx)

    # Search phase (1.0s later): vx=0, rotates left (positive) toward last seen side
    vx2, om2, _, alert2 = ctrl.step_person([], now=11.0)
    assert vx2 == 0.0 and om2 > 0 and alert2 is None

    # Full stop after lost_search_sec (5.0s later): stops and emits alert once
    vx3, om3, _, alert3 = ctrl.step_person([], now=15.0)
    assert vx3 == 0.0 and om3 == 0.0 and alert3 is not None
    _, _, _, alert4 = ctrl.step_person([], now=15.5)
    assert alert4 is None


def test_detect_laser_spot_synthetic():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a bright red dot at (200, 180) in BGR
    cv2.circle(frame, (200, 180), 6, (0, 0, 255), -1)
    spot = avf.detect_laser_spot(frame)
    assert spot is not None
    u, v, strength = spot
    assert abs(u - 200.0) < 1.5
    assert abs(v - 180.0) < 1.5
    assert strength > 0.9

    vx, omega = avf.compute_laser_velocity(spot, avf.FollowConfig())
    assert omega > 0, "Laser dot on left should turn CCW"
    assert vx > 0, "Laser dot high in frame (far) should move forward"

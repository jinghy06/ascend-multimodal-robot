import threading
import time

import numpy as np
import pytest
from pynput.keyboard import Key, KeyCode

import atlas_leader_host as host_mod
import pc_leader_teleop as pc_mod


def test_map_leader_to_lekiwi_arm_clamps_and_rejects_nan():
    raw = {
        "shoulder_pan.pos": 150.0,
        "shoulder_lift.pos": -120.0,
        "elbow_flex.pos": 25.0,
        "wrist_flex.pos": -10.0,
        "wrist_roll.pos": 5.0,
        "gripper.pos": -15.0,
    }
    mapped = host_mod.map_leader_to_lekiwi_arm(raw)
    assert mapped == {
        "arm_shoulder_pan.pos": 100.0,
        "arm_shoulder_lift.pos": -100.0,
        "arm_elbow_flex.pos": 25.0,
        "arm_wrist_flex.pos": -10.0,
        "arm_wrist_roll.pos": 5.0,
        "arm_gripper.pos": 0.0,
    }

    bad = dict(raw, **{"elbow_flex.pos": float("nan")})
    with pytest.raises(ValueError, match="Non-finite"):
        host_mod.map_leader_to_lekiwi_arm(bad)


def test_soft_start_arm_follower_ramps_smoothly_without_snapping():
    init_obs = {j: 0.0 for j in host_mod.ARM_JOINTS}
    follower = host_mod.SoftStartArmFollower(init_obs, soft_start_sec=1.0, max_arm_delta=5.0)
    follower.reset(init_obs, now=10.0)
    desired = {j: 60.0 for j in host_mod.ARM_JOINTS}

    # At t=10.1 (alpha=0.1), target blends to 6.0, capped by max_arm_delta=5.0
    step1 = follower.compute_targets(desired, init_obs, now=10.1)
    assert step1["arm_shoulder_pan.pos"] == pytest.approx(5.0)

    # After soft start completes (t=11.0), per-step delta cap still protects servos
    step2 = follower.compute_targets(desired, step1, now=11.0)
    assert step2["arm_shoulder_pan.pos"] == pytest.approx(10.0)


def test_wireless_teleop_state_watchdog_and_seq_dedup():
    state = host_mod.WirelessTeleopState(watchdog_sec=0.2)
    arm = {j: 10.0 for j in host_mod.ARM_JOINTS}
    pkt = {"seq": 1, "arm": arm, "base": {"x.vel": 0.5, "y.vel": 0.0, "theta.vel": 100.0}, "demo_action": "toggle", "task": "pick"}

    assert state.apply_packet(pkt, now=1.0) is True
    # Duplicate seq ignored
    assert state.apply_packet(pkt, now=1.05) is False
    assert state.pop_demo_actions() == [("toggle", "pick")]

    got_arm, got_base, stale, _ = state.snapshot(now=1.1)
    assert stale is False
    assert got_arm["arm_shoulder_pan.pos"] == 10.0
    assert got_base["x.vel"] == pytest.approx(0.22)
    assert got_base["theta.vel"] == pytest.approx(65.0)

    # Past watchdog -> stale & zero velocity
    _, timed_out_base, stale_after, _ = state.snapshot(now=1.35)
    assert stale_after is True
    assert timed_out_base == {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


def test_leader_base_keyboard_state_bindings():
    kb = pc_mod.LeaderBaseKeyboardState()
    kb.on_press(KeyCode.from_char("W"))
    kb.on_press(Key.shift_l)
    pressed, stopped = kb.snapshot()
    assert stopped is False
    assert pressed == {"w", "boost"}

    # Rising-edge hotkeys
    kb.on_press(KeyCode.from_char("r"))
    kb.on_press(KeyCode.from_char("r"))
    kb.on_press(Key.backspace)
    assert kb.pop_demo_actions() == ["toggle", "discard"]

    # Space locks base; Esc exits
    kb.on_press(Key.space)
    pressed2, stopped2 = kb.snapshot()
    assert stopped2 is True
    assert "w" not in pressed2

    kb.on_press(Key.esc)
    assert kb.running is False


def test_end_to_end_udp_loopback_records_and_discards(tmp_path):
    started_ev = threading.Event()
    frame_ev = threading.Event()

    class FakeDataset:
        def __init__(self):
            self.frames = []
            self.saved = False
            self.cleared = False

        def add_frame(self, frame):
            self.frames.append(frame)
            frame_ev.set()

        def save_episode(self):
            self.saved = True

        def clear_episode_buffer(self):
            self.cleared = True

    created = []

    def ds_factory(**kwargs):
        ds = FakeDataset()
        created.append(ds)
        started_ev.set()
        return ds

    host = host_mod.AtlasLeaderHost(
        bind_host="127.0.0.1",
        udp_port=0,
        work_dir=tmp_path,
        rate=40.0,
        record_fps=10,
        tts_udp_port=0,
        dry_run=True,
        dataset_factory=ds_factory,
    )
    t = threading.Thread(target=host.run, kwargs={"max_seconds": 12.0, "exit_on_stop": True}, daemon=True)
    t.start()

    class MockLeader:
        def get_action(self):
            return {
                "shoulder_pan.pos": 12.0,
                "shoulder_lift.pos": -10.0,
                "elbow_flex.pos": 18.0,
                "wrist_flex.pos": -5.0,
                "wrist_roll.pos": 2.0,
                "gripper.pos": 40.0,
            }

    kb = pc_mod.LeaderBaseKeyboardState()

    def drive_hotkeys():
        time.sleep(0.1)
        kb.on_press(KeyCode.from_char("r"))
        kb.on_release(KeyCode.from_char("r"))
        started_ev.wait(timeout=10.0)
        frame_ev.wait(timeout=5.0)
        kb.on_press(Key.backspace)
        kb.on_release(Key.backspace)
        time.sleep(0.2)
        kb.on_press(Key.esc)

    threading.Thread(target=drive_hotkeys, daemon=True).start()
    summary = pc_mod.run_sender_loop(
        MockLeader(),
        kb,
        atlas_ip="127.0.0.1",
        udp_port=host.bound_port,
        rate=30.0,
        task="pick up the orange",
        max_seconds=12.0,
        quiet=True,
    )
    t.join(timeout=3.0)

    assert summary["sent"] >= 10
    assert summary["acks"] >= 1
    assert len(created) == 1
    assert len(created[0].frames) >= 1
    assert created[0].cleared is True
    assert created[0].saved is False

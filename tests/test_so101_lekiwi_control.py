from pathlib import Path

import pytest
from pynput.keyboard import Key, KeyCode

import so101_lekiwi_control as control


ARM_OBSERVATION = {
    "arm_shoulder_pan.pos": 0.0,
    "arm_shoulder_lift.pos": 0.0,
    "arm_elbow_flex.pos": 0.0,
    "arm_wrist_flex.pos": 0.0,
    "arm_wrist_roll.pos": 0.0,
    "arm_gripper.pos": 0.0,
}


def key(char):
    return KeyCode.from_char(char)


@pytest.mark.parametrize(
    ("pressed", "expected"),
    [
        ({"w"}, (0.1, 0.0, 0.0)),
        ({"s"}, (-0.1, 0.0, 0.0)),
        ({"a"}, (0.0, 0.1, 0.0)),
        ({"d"}, (0.0, -0.1, 0.0)),
        ({"z"}, (0.0, 0.0, 30.0)),
        ({"x"}, (0.0, 0.0, -30.0)),
        ({"w", "a", "z"}, (0.1, 0.1, 30.0)),
        ({"w", "s", "a", "d", "z", "x"}, (0.0, 0.0, 0.0)),
    ],
)
def test_base_velocity_directions(pressed, expected):
    action = control.base_velocity(pressed, False, 0.1, 30.0)

    assert action == {"x.vel": expected[0], "y.vel": expected[1], "theta.vel": expected[2]}


def test_base_stop_lock_overrides_held_keys():
    assert control.base_velocity({"w", "z"}, True, 0.1, 30.0) == {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }


def test_base_boost_doubles_speed_but_stop_has_priority():
    assert control.base_velocity({"w", "z", "boost"}, False, 0.1, 30.0) == {
        "x.vel": 0.2,
        "y.vel": 0.0,
        "theta.vel": 60.0,
    }
    assert control.base_velocity({"w", "z", "boost"}, True, 0.1, 30.0) == {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }


def test_keyboard_stop_lock_clears_base_keys_until_new_movement():
    state = control.KeyboardState()
    state.on_press(key("W"))
    state.on_press(key("j"))

    state.on_press(Key.space)
    pressed, stopped = state.snapshot()

    assert stopped is True
    assert "w" not in pressed
    assert "j" in pressed

    state.on_press(key("w"))
    pressed, stopped = state.snapshot()
    assert stopped is True
    assert "w" not in pressed

    state.on_press(key("d"))
    pressed, stopped = state.snapshot()
    assert stopped is False
    assert pressed == {"j", "d"}


def test_keyboard_release_and_escape():
    state = control.KeyboardState()
    state.on_press(key("W"))
    state.on_release(key("w"))
    assert state.snapshot() == (set(), False)

    state.on_press(Key.esc)
    assert state.running is False


def test_arm_targets_cover_all_six_joints():
    pressed = {"j", "i", "u", "e", Key.up, Key.left}

    targets = control.arm_targets(ARM_OBSERVATION, pressed, 0.8)

    assert targets == {
        "arm_shoulder_pan.pos": 0.8,
        "arm_shoulder_lift.pos": 0.8,
        "arm_elbow_flex.pos": 0.8,
        "arm_wrist_flex.pos": 0.8,
        "arm_wrist_roll.pos": 0.8,
        "arm_gripper.pos": 0.8,
    }


def test_arm_targets_hold_and_cancel_opposites():
    targets = control.arm_targets(
        ARM_OBSERVATION,
        {"j", "l", "i", "k", "u", "o", "e", "q", Key.up, Key.down, Key.left, Key.right},
        0.8,
    )

    assert targets == ARM_OBSERVATION


def test_arm_targets_clamp_normalized_ranges():
    observation = dict.fromkeys(ARM_OBSERVATION, 99.6)

    high = control.arm_targets(observation, {"j", "i", "u", "e", Key.up, Key.left}, 0.8)
    low = control.arm_targets(dict.fromkeys(ARM_OBSERVATION, -99.6), {"l", "k", "o", "q", Key.down, Key.right}, 0.8)

    assert high == {
        "arm_shoulder_pan.pos": 100.0,
        "arm_shoulder_lift.pos": 100.0,
        "arm_elbow_flex.pos": 100.0,
        "arm_wrist_flex.pos": 100.0,
        "arm_wrist_roll.pos": 100.0,
        "arm_gripper.pos": 100.0,
    }
    assert low == {
        "arm_shoulder_pan.pos": -100.0,
        "arm_shoulder_lift.pos": -100.0,
        "arm_elbow_flex.pos": -100.0,
        "arm_wrist_flex.pos": -100.0,
        "arm_wrist_roll.pos": -100.0,
        "arm_gripper.pos": 0.0,
    }


class FakeControls:
    def __init__(self):
        self.running = True

    def snapshot(self):
        self.running = False
        return set(), False


class SequenceControls:
    def __init__(self, snapshots):
        self.running = True
        self.snapshots = list(snapshots)

    def snapshot(self):
        snapshot = self.snapshots.pop(0)
        if not self.snapshots:
            self.running = False
        return snapshot


class FakeRobot:
    def __init__(self, fail=False):
        self.fail = fail
        self.actions = []
        self.stop_calls = 0
        self.disconnect_calls = 0

    def get_observation(self):
        return dict(ARM_OBSERVATION)

    def send_action(self, action):
        self.actions.append(action)
        if self.fail:
            raise RuntimeError("simulated failure")
        return action

    def stop_base(self):
        self.stop_calls += 1

    def disconnect(self):
        self.disconnect_calls += 1


def test_control_loop_sends_zero_velocity_and_cleans_up():
    robot = FakeRobot()
    controls = FakeControls()

    control.run_control_loop(robot, controls, 0.1, 30.0, 0.8, 20.0)

    assert robot.actions == [{**ARM_OBSERVATION, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}]
    assert robot.stop_calls == 1
    assert robot.disconnect_calls == 1


def test_control_loop_forwards_observation_and_actual_action_to_demo():
    robot = FakeRobot()
    controls = FakeControls()
    frames = []

    class Demo:
        def record(self, observation, action):
            frames.append((observation, action))

    control.run_control_loop(robot, controls, 0.1, 30.0, 0.8, 20.0, demo=Demo())

    assert frames == [(ARM_OBSERVATION, {**ARM_OBSERVATION, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})]


def test_control_loop_accumulates_targets_and_holds_after_key_release():
    robot = FakeRobot()
    controls = SequenceControls([({"i"}, False), ({"i"}, False), (set(), False)])

    control.run_control_loop(robot, controls, 0.1, 30.0, 0.8, 20.0)

    assert [action["arm_shoulder_lift.pos"] for action in robot.actions] == [0.8, 1.6, 1.6]


def test_control_loop_cleans_up_when_action_fails():
    robot = FakeRobot(fail=True)

    with pytest.raises(RuntimeError, match="simulated failure"):
        control.run_control_loop(robot, FakeControls(), 0.1, 30.0, 0.8, 20.0)

    assert robot.stop_calls == 1
    assert robot.disconnect_calls == 1


def test_missing_calibration_fails_before_connection(tmp_path):
    class CalibrationOnlyRobot:
        calibration_fpath = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="calibration"):
        control.ensure_calibration(CalibrationOnlyRobot(), False)

    control.ensure_calibration(CalibrationOnlyRobot(), True)
    Path(CalibrationOnlyRobot.calibration_fpath).touch()
    control.ensure_calibration(CalibrationOnlyRobot(), False)


def test_parser_defaults_to_com9_without_calibration():
    args = control.parse_args([])

    assert args.port == "COM9"
    assert args.id == "my_awesome_kiwi"
    assert args.calibrate is False


def test_bus_configuration_writes_use_retries():
    class FakeBus:
        def __init__(self):
            self.calls = []

        def write(self, name, motor, value, num_retry=0):
            self.calls.append((name, motor, value, num_retry))

    class FakeRobot:
        bus = FakeBus()

    robot = FakeRobot()
    control.configure_bus_retries(robot, retries=3)
    robot.bus.write("Lock", 2, 1)
    robot.bus.write("Operating_Mode", 2, 0, num_retry=0)

    assert robot.bus.calls == [("Lock", 2, 1, 3), ("Operating_Mode", 2, 0, 3)]


def test_robot_config_avoids_lerobot_lekiwi_relative_target_key_mismatch():
    config = control.make_robot_config("COM9", "my_awesome_kiwi")

    assert config.port == "COM9"
    assert config.id == "my_awesome_kiwi"
    assert config.cameras == {}
    assert config.max_relative_target is None


def test_evdev_decoder_maps_press_repeat_release_and_arrows():
    import struct

    presses = []
    releases = []
    events = [
        struct.pack("@llHHI", 0, 0, 1, 22, 1),
        struct.pack("@llHHI", 0, 0, 1, 22, 2),
        struct.pack("@llHHI", 0, 0, 1, 22, 0),
        struct.pack("@llHHI", 0, 0, 1, 103, 1),
        struct.pack("@llHHI", 0, 0, 1, 103, 0),
    ]

    control.decode_evdev_events(b"".join(events), presses.append, releases.append)

    assert presses == ["u", "u", "up"]
    assert releases == ["u", "up"]


def test_evdev_decoder_ignores_unmapped_and_non_key_events():
    import struct

    presses = []
    releases = []
    events = [
        struct.pack("@llHHI", 0, 0, 0, 22, 1),
        struct.pack("@llHHI", 0, 0, 1, 999, 1),
        struct.pack("@llHHI", 0, 0, 1, 22, 3),
    ]

    control.decode_evdev_events(b"".join(events), presses.append, releases.append)

    assert presses == []
    assert releases == []


def test_parser_defaults_to_atlas_keyboard_device():
    args = control.parse_args([])

    assert args.keyboard_device == "/dev/input/event0"


def test_parser_uses_linux_serial_default(monkeypatch):
    monkeypatch.setattr(control.sys, "platform", "linux")

    args = control.parse_args([])

    assert args.port == "/dev/ttyACM0"


def test_linux_default_prefers_ch343_by_id(monkeypatch):
    monkeypatch.setattr(control.sys, "platform", "linux")
    monkeypatch.setattr(control.glob, "glob", lambda pattern: ["/dev/serial/by-id/ch343"])

    args = control.parse_args([])

    assert args.port == "/dev/serial/by-id/ch343"


def test_analog_scales_apply_to_base_velocity_and_arm_targets():
    vel = control.base_velocity({"w", "z"}, False, 0.1, 30.0, scales={"w": 0.5, "z": 0.25})
    assert vel == {"x.vel": pytest.approx(0.05), "y.vel": 0.0, "theta.vel": pytest.approx(7.5)}

    targets = control.arm_targets(ARM_OBSERVATION, {"j", "i"}, 1.0, scales={"j": 0.5})
    assert targets["arm_shoulder_pan.pos"] == pytest.approx(0.5)
    assert targets["arm_shoulder_lift.pos"] == pytest.approx(1.0)


def test_control_loop_skips_observation_when_demo_does_not_want_it():
    class CountingRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.obs_calls = 0

        def get_observation(self):
            self.obs_calls += 1
            return super().get_observation()

    class SelectiveControls:
        def __init__(self):
            self.step = 0
            self.running = True

        def snapshot(self):
            self.step += 1
            if self.step >= 3:
                self.running = False
            return {"w"}, False

        def analog_scales(self):
            return {"w": 0.5}

    class SelectiveDemo:
        def __init__(self):
            self.checks = 0
            self.recorded = []

        def wants_observation(self):
            self.checks += 1
            return self.checks == 2

        def record(self, observation, action):
            self.recorded.append((observation, action))

    robot = CountingRobot()
    demo = SelectiveDemo()
    control.run_control_loop(robot, SelectiveControls(), 0.1, 30.0, 0.8, 100.0, demo=demo)

    # 1 initial get_observation() + 1 when demo.wants_observation() returned True
    assert robot.obs_calls == 2
    assert len(demo.recorded) == 3
    assert demo.recorded[0][1]["x.vel"] == pytest.approx(0.05)

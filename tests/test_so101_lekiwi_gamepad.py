import struct

import pytest

import so101_lekiwi_gamepad as gamepad


BUTTON = 1
AXIS = 2


@pytest.mark.parametrize(
    ("axis", "value", "expected"),
    [
        (0, -32767, {"a"}),
        (0, 32767, {"d"}),
        (1, -32767, {"w"}),
        (1, 32767, {"s"}),
        (3, -32767, {"z"}),
        (3, 32767, {"x"}),
    ],
)
def test_sticks_control_base(axis, value, expected):
    state = gamepad.GamepadState()
    state.handle_event(AXIS, axis, value)

    assert state.snapshot() == (expected, False)


def test_stick_deadzone_is_neutral():
    state = gamepad.GamepadState()
    for axis in (0, 1, 3):
        state.handle_event(AXIS, axis, 12000)

    assert state.snapshot() == (set(), False)


@pytest.mark.parametrize(
    ("axis", "value", "expected"),
    [
        (4, -32767, {"i"}),
        (4, 32767, {"k"}),
        (6, -32767, {"j"}),
        (6, 32767, {"l"}),
        (7, -32767, {"u"}),
        (7, 32767, {"o"}),
        (2, 32767, {"left"}),
        (5, 32767, {"right"}),
        (2, -32767, set()),
        (5, -32767, set()),
    ],
)
def test_axes_control_arm(axis, value, expected):
    state = gamepad.GamepadState()
    state.handle_event(AXIS, axis, value)

    assert state.snapshot() == (expected, False)


@pytest.mark.parametrize(
    ("button", "expected"),
    [
        (0, "e"),
        (1, "q"),
        (4, "up"),
        (5, "down"),
    ],
)
def test_buttons_control_gripper_and_wrist(button, expected):
    state = gamepad.GamepadState()
    state.handle_event(BUTTON, button, 1)
    assert state.snapshot()[0] == {expected}

    state.handle_event(BUTTON, button, 0)
    assert state.snapshot()[0] == set()


def test_start_stops_base_until_all_base_controls_return_neutral():
    state = gamepad.GamepadState()
    state.handle_event(AXIS, 1, -32767)
    state.handle_event(BUTTON, 0, 1)
    state.handle_event(BUTTON, 7, 1)

    assert state.snapshot() == ({"e"}, True)

    state.handle_event(AXIS, 1, -32767)
    assert state.snapshot() == ({"e"}, True)

    state.handle_event(AXIS, 1, 0)
    assert state.snapshot() == ({"e"}, False)

    state.handle_event(AXIS, 1, -32767)
    assert state.snapshot() == ({"w", "e"}, False)


def test_back_button_exits():
    state = gamepad.GamepadState()
    state.handle_event(BUTTON, 6, 1)

    assert state.running is False


def test_decoder_handles_init_events_and_ignores_partial_data():
    state = gamepad.GamepadState()
    data = b"".join(
        [
            struct.pack("IhBB", 0, -32767, AXIS | 0x80, 1),
            struct.pack("IhBB", 0, 1, BUTTON, 0),
            b"\x00",
        ]
    )

    gamepad.decode_joystick_events(data, state)

    assert state.snapshot() == ({"w", "e"}, False)


def xbox_report(buttons=0, left_trigger=0, right_trigger=0, left_x=0, left_y=0, right_x=0, right_y=0):
    data = bytearray(20)
    data[1] = 20
    struct.pack_into("<H", data, 2, buttons)
    data[4] = left_trigger
    data[5] = right_trigger
    struct.pack_into("<hhhh", data, 6, left_x, left_y, right_x, right_y)
    return data


def test_xbox_report_decodes_buttons_and_releases():
    state = gamepad.GamepadState()
    buttons = 0x1000 | 0x2000 | 0x0100 | 0x0200

    gamepad.decode_xbox360_report(xbox_report(buttons), state)
    assert state.snapshot()[0] == {"e", "q", "up", "down"}

    gamepad.decode_xbox360_report(xbox_report(), state)
    assert state.snapshot()[0] == set()


def test_xbox_report_decodes_sticks_with_north_east_positive():
    state = gamepad.GamepadState()
    report = xbox_report(left_x=-32767, left_y=32767, right_x=32767, right_y=32767)

    gamepad.decode_xbox360_report(report, state)

    assert state.snapshot()[0] == {"a", "w", "x", "i"}


def test_xbox_report_decodes_dpad_and_triggers():
    state = gamepad.GamepadState()
    report = xbox_report(buttons=0x0005, left_trigger=255, right_trigger=128)

    gamepad.decode_xbox360_report(report, state)

    assert state.snapshot()[0] == {"j", "u", "left", "right"}


def test_xbox_report_decodes_start_back_and_ignores_short_data():
    state = gamepad.GamepadState()
    gamepad.decode_xbox360_report(b"\x00\x14", state)
    assert state.snapshot() == (set(), False)

    gamepad.decode_xbox360_report(xbox_report(buttons=0x0010), state)
    assert state.snapshot()[1] is True

    gamepad.decode_xbox360_report(xbox_report(buttons=0x0020), state)
    assert state.running is False


def test_parser_defaults_to_xbox_usb_backend():
    args = gamepad.parse_args([])

    assert args.input_backend == "usb"


def test_gamepad_default_port_uses_ch343_by_id(monkeypatch):
    monkeypatch.setattr(gamepad, "find_ch343_port", lambda: "/dev/serial/by-id/ch343")

    args = gamepad.parse_args([])

    assert args.port == "/dev/serial/by-id/ch343"


def test_xbox_x_button_enables_boost():
    state = gamepad.GamepadState()
    gamepad.decode_xbox360_report(xbox_report(buttons=0x4000), state)
    assert state.snapshot()[0] == {"boost"}

    gamepad.decode_xbox360_report(xbox_report(), state)
    assert state.snapshot()[0] == set()


def test_parser_defaults_to_30hz_decoupled_10fps():
    args = gamepad.parse_args([])

    assert args.rate == 30.0
    assert args.arm_step == 0.7
    assert args.record_fps == 10
    assert args.demo_task == "pick up the orange"
    assert args.tts_udp_port == 5007


def test_analog_stick_scales_proportionally_and_dpad_is_full_scale():
    state = gamepad.GamepadState(deadzone=12000)
    # Halfway beyond deadzone (~22383): should yield 0 < scale < 1
    state.handle_event(AXIS, 1, -22383)
    scales = state.analog_scales()
    assert 0.2 < scales["w"] < 0.7

    # Full stick & D-pad: should yield 1.0
    state.handle_event(AXIS, 1, -32767)
    state.handle_event(AXIS, 6, 32767)
    scales = state.analog_scales()
    assert scales["w"] == pytest.approx(1.0)
    assert scales["l"] == pytest.approx(1.0)


def test_xbox_y_button_rising_edge_emits_toggle_and_start_y_emits_discard():
    state = gamepad.GamepadState()
    gamepad.decode_xbox360_report(xbox_report(buttons=0x8000), state)
    gamepad.decode_xbox360_report(xbox_report(buttons=0x8000), state)
    assert state.pop_demo_actions() == ["toggle"]
    assert state.pop_demo_actions() == []

    gamepad.decode_xbox360_report(xbox_report(buttons=0x0010), state)
    gamepad.decode_xbox360_report(xbox_report(buttons=0x0010 | 0x8000), state)
    assert state.pop_demo_actions() == ["discard"]


def test_handle_gamepad_demo_action_toggles_and_discards(tmp_path):
    import json

    assert gamepad.handle_gamepad_demo_action("toggle", tmp_path, task="pick up cup", tts_port=0) == "started"
    cmd = json.loads((tmp_path / "demo_command.json").read_text(encoding="utf-8"))
    assert cmd["active"] is True
    assert cmd["task"] == "pick up cup"

    assert gamepad.handle_gamepad_demo_action("toggle", tmp_path, tts_port=0) == "stopping"
    cmd = json.loads((tmp_path / "demo_command.json").read_text(encoding="utf-8"))
    assert cmd["active"] is False

    (tmp_path / "demo_status.json").write_text(json.dumps({"recording": True}), encoding="utf-8")
    assert gamepad.handle_gamepad_demo_action("discard", tmp_path, tts_port=0) == "discarding"
    cmd = json.loads((tmp_path / "demo_command.json").read_text(encoding="utf-8"))
    assert cmd["active"] is False
    assert cmd["discard"] is True

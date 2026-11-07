import pytest

from feedback_policy import make_style_request, render_feedback


@pytest.mark.parametrize("invalid", ["已经拿到了", "B。马上去拿", "Z", " B", None])
def test_invalid_model_reply_cannot_override_failure(invalid):
    reply = render_feedback("grasp_failed_not_held_stopped", lambda _: invalid)
    assert reply.source == "local"
    assert reply.text == "还没拿到，刚才没夹稳。"


def test_qwen_selects_natural_variant_but_cannot_choose_target():
    reply = render_feedback("two_red_blocks_left_right", lambda _: "B")
    assert reply.source == "qwen"
    assert "左边的还是右边的" in reply.text


def test_stop_and_unknown_never_call_model():
    def forbidden(_):
        raise AssertionError("should not invoke model")

    assert render_feedback("stopped_confirmed", forbidden).text == "停好了。"
    assert render_feedback("grasp_accepted", forbidden).source == "unknown"


def test_exclude_previous_variant_and_handle_timeout():
    allowed, req = make_style_request("speech_unclear", previous="A")
    assert "A" not in allowed and req["max_tokens"] == 1
    assert req["grammar"] == 'root ::= "B" | "C"'

    def timeout(_):
        raise TimeoutError()

    assert render_feedback("speech_unclear", timeout, previous="A").variant == "B"

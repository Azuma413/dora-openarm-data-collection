"""Headset A/B buttons: B outside an episode quits only when held."""

from dora_openarm_data_collection_ui.main import HeadsetButtons


def press(buttons, button, collecting, times):
    """Press `button` at each time and return what each press asks the UI to do."""
    return [buttons.on_button(button, True, collecting, t) for t in times]


def test_a_starts_and_finishes_on_the_rising_edge_only():
    """A acts once per press, not once per frame."""
    b = HeadsetButtons()
    assert press(b, "button_a", False, [0.0, 0.01, 0.02]) == ["start", None, None]
    b.on_button("button_a", False, True, 0.03)
    assert press(b, "button_a", True, [0.04]) == ["success"]


def test_a_short_b_press_outside_an_episode_does_not_quit():
    """A tap on B between episodes is harmless."""
    b = HeadsetButtons()
    assert press(b, "button_b", False, [0.0, 0.5, 1.9]) == [None, None, None]
    assert b.on_button("button_b", False, False, 1.95) is None
    # A new press starts a new hold from zero.
    assert press(b, "button_b", False, [2.0, 3.9]) == [None, None]


def test_holding_b_for_two_seconds_quits_once():
    """Quit needs a deliberate 2 s hold, and fires once."""
    b = HeadsetButtons()
    assert press(b, "button_b", False, [0.0, 1.0, 2.0, 2.5]) == [
        None,
        None,
        "quit",
        None,
    ]


def test_b_during_an_episode_fails_and_does_not_run_on_into_a_quit():
    """The press that fails an episode cannot also quit."""
    b = HeadsetButtons()
    # The episode ends on the first frame; the operator keeps holding B.
    assert press(b, "button_b", True, [0.0]) == ["fail"]
    assert press(b, "button_b", False, [0.5, 2.5, 4.0]) == [None, None, None]

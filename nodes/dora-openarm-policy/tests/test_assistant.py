"""The voice assistant with a scripted backend: no OpenAI, no dora."""

import queue
from pathlib import Path

import cv2
import numpy as np
import pytest

from dora_openarm_policy import web
from dora_openarm_policy.assistant import EXECUTE, LOCATE, SAY_STOPPED, Assistant
from dora_openarm_policy.presets import Presets, annotate

PRESETS_YAML = """
image_size: [1280, 720]
step_px: 30
margin_px: 40
presets:
  - {id: 1, name: 左肩の上, bbox: [600, 300, 700, 400]}
  - {id: 2, name: 左肩の下, bbox: [600, 450, 700, 550]}
"""


class Script:
    """Answers decide() from a list and records every call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.spoken = []

    def transcribe(self, wav):
        return "左肩に貼って"

    def speak(self, text):
        self.spoken.append(text)
        return b"mp3"

    def decide(self, instructions, prompt, image_jpeg, schema, name):
        self.calls.append((name, prompt))
        return self.answers.pop(0)


@pytest.fixture
def presets(tmp_path):
    path = tmp_path / "presets.yaml"
    path.write_text(PRESETS_YAML, encoding="utf-8")
    return Presets.load(path)


def make(presets, answers, status=None):
    shared = web.Shared()
    _ok, jpeg = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))
    shared.head_jpeg = shared.snapshot_jpeg = jpeg.tobytes()
    shared.status = status or {"mode": "passthrough", "can_start": None, "bbox": None}
    backend = Script(answers)
    lines = []
    a = Assistant(shared, presets, backend, lines.append, speech_timeout_s=5.0)
    return a, shared, backend


def requests(shared):
    out = []
    while True:
        try:
            out.append(shared.requests.get_nowait())
        except queue.Empty:
            return out


def test_it_asks_until_the_place_is_unique_then_sets_the_bbox_and_starts_after_speaking(presets):
    a, shared, backend = make(
        presets,
        [
            {"reply": "肩の上の方ですか、下の方ですか。", "decision": "ask", "preset_id": None, "target": ""},
            {
                "reply": "左肩の上の方に貼ります。",
                "decision": "decided",
                "preset_id": 1,
                "target": "左肩の上",
            },
        ],
    )
    a.handle("voice", b"wav")
    assert a.phase == LOCATE and backend.spoken == ["肩の上の方ですか、下の方ですか。"]
    assert requests(shared) == []
    a.handle("text", "上の方")
    assert a.phase == EXECUTE and a.preset_id == 1
    assert requests(shared) == [("bbox", [600 / 1280, 300 / 720, 700 / 1280, 400 / 720])]
    # The second call saw the whole conversation so far.
    assert "肩の上の方ですか" in backend.calls[1][1] and "左肩に貼って" in backend.calls[1][1]
    # Not before the reply has been played.
    a.poll(0.0)
    assert requests(shared) == []
    a.handle("speech_done", a.speech["id"])
    assert requests(shared) == [("action", "start")]


def test_start_waits_for_the_intro(presets):
    a, shared, backend = make(
        presets,
        [{"reply": "貼ります。", "decision": "decided", "preset_id": 2, "target": "左肩の下"}],
        status={
            "mode": "passthrough",
            "can_start": "the left arm's intro is not done (waiting)",
            "bbox": None,
        },
    )
    a.handle("text", "左肩の下")
    requests(shared)
    a.handle("speech_done", a.speech["id"])
    assert requests(shared) == [] and backend.spoken[-1] == "湿布を受け取ったら、すぐに始めます。"
    shared.status = {"mode": "passthrough", "can_start": None, "bbox": None}
    a.poll(1.0)
    assert requests(shared) == [("action", "start")]


def test_a_correction_moves_the_bbox_in_steps_within_range_and_restarts(presets):
    bbox = presets.normalized([600, 450, 700, 550])
    a, shared, backend = make(
        presets,
        [
            {
                "reply": "少し上にずらします。",
                "action": "adjust",
                "dx_steps": 0,
                "dy_steps": -2,
                "preset_id": None,
            },
            {
                "reply": "右にずらします。",
                "action": "adjust",
                "dx_steps": 10,
                "dy_steps": 0,
                "preset_id": None,
            },
        ],
        status={"mode": "stopped", "can_start": None, "bbox": bbox},
    )
    a.phase = EXECUTE
    a.handle("text", "もっと上")
    ((kind, moved),) = requests(shared)
    assert kind == "bbox"
    np.testing.assert_allclose(presets.pixels(moved), [600, 390, 700, 490])
    a.handle("speech_done", a.speech["id"])
    assert requests(shared) == [("action", "start")]
    # Far right: clamped to the presets' extent plus the margin (center x 650 + 40).
    a.handle("text", "ずっと右")
    ((_, moved),) = requests(shared)
    px = presets.pixels(moved)
    assert (px[0] + px[2]) / 2 == pytest.approx(690)
    assert "これ以上は動かせない" in backend.spoken[-1]


def test_stop_and_noise_keep_the_robot_stopped(presets):
    bbox = presets.normalized([600, 450, 700, 550])
    a, shared, backend = make(
        presets,
        [
            {
                "reply": "止まったままにします。",
                "action": "stop",
                "dx_steps": 0,
                "dy_steps": 0,
                "preset_id": None,
            }
        ],
        status={"mode": "stopped", "can_start": None, "bbox": bbox},
    )
    a.phase = EXECUTE
    a.handle("text", "止まって")
    a.handle("speech_done", a.speech["id"])
    assert requests(shared) == []
    # Nothing usable after a voice stop: say so, stay stopped, no VLM call.
    backend.transcribe = lambda wav: "ご視聴ありがとうございました"
    a.handle("voice", b"wav")
    assert backend.spoken[-1] == SAY_STOPPED and len(backend.calls) == 1
    assert requests(shared) == []


def test_without_a_key_it_reports_and_does_nothing(presets):
    shared = web.Shared()
    a = Assistant(shared, presets, None, lambda m: None, setup_error="OPENAI_API_KEY is not set")
    a.handle("text", "左肩")
    assert shared.assistant_status["enabled"] is False
    assert "OPENAI_API_KEY" in shared.assistant_status["error"]
    assert requests(shared) == []


def test_presets_load_annotate_and_ship_the_repo_file(presets):
    _ok, jpeg = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))
    out = cv2.imdecode(np.frombuffer(annotate(jpeg.tobytes(), presets, [600, 300, 700, 400]), np.uint8), 1)
    assert out.shape == (360, 640, 3) and out.sum() > 0
    repo = Presets.load(Path(__file__).parents[3] / "configs" / "patch_presets.yaml")
    assert len(repo.items) >= 1 and repo.image_size == (1280, 720)


def _wav(samples):
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.asarray(samples, np.int16).tobytes())
    return buf.getvalue()


def _peak(wav):
    import io
    import wave

    with wave.open(io.BytesIO(wav)) as w:
        return np.abs(np.frombuffer(w.readframes(w.getnframes()), np.int16)).max() / 32768


def test_quiet_speech_is_brought_up_and_loud_speech_down():
    from dora_openarm_policy.assistant import NORMALIZE_PEAK, normalize_wav

    t = np.arange(16000) / 16000
    quiet = _wav(np.sin(2 * np.pi * 300 * t) * 32768 * 10 ** (-45 / 20))  # -45 dBFS, like the lab mic
    out, gain = normalize_wav(quiet)
    assert gain == pytest.approx(NORMALIZE_PEAK / 10 ** (-45 / 20), rel=0.01)
    assert _peak(out) == pytest.approx(NORMALIZE_PEAK, abs=0.02)
    loud = _wav(np.sin(2 * np.pi * 300 * t) * 32000)
    out, gain = normalize_wav(loud)
    assert gain < 1.0 and _peak(out) == pytest.approx(NORMALIZE_PEAK, abs=0.02)
    # Near silence: capped, not blown up to full scale.
    _, gain = normalize_wav(_wav(np.full(16000, 2)))
    assert gain == 200.0
    # Not a WAV: untouched.
    assert normalize_wav(b"not a wav") == (b"not a wav", 1.0)


def test_the_backend_gets_the_normalized_utterance(presets):
    a, _shared, backend = make(
        presets, [{"reply": "どこですか。", "decision": "ask", "preset_id": None, "target": ""}]
    )
    seen = []
    backend.transcribe = lambda wav: seen.append(_peak(wav)) or "左肩"
    t = np.arange(16000) / 16000
    a.handle("voice", _wav(np.sin(2 * np.pi * 300 * t) * 300))
    assert seen and seen[0] > 0.6


def test_noise_too_quiet_to_be_speech_is_dropped_before_transcription(presets):
    bbox = presets.normalized([600, 450, 700, 550])
    a, shared, backend = make(
        presets, [], status={"mode": "stopped", "reason": "voice", "can_start": None, "bbox": bbox}
    )
    a.phase = EXECUTE
    heard = []
    backend.transcribe = lambda wav: heard.append(wav) or "続けて"
    t = np.arange(16000) / 16000
    noise = _wav(np.sin(2 * np.pi * 300 * t) * 32768 * 10 ** (-52 / 20))  # motors, -52 dBFS
    a.handle("voice", noise)
    # Never transcribed, nothing said: the robot the noise stopped carries on.
    assert heard == [] and backend.spoken == []
    assert requests(shared) == [("action", "start")]
    # Stopped by the UI, not by the voice: noise does not restart it.
    shared.status = {"mode": "stopped", "reason": "UI", "can_start": None, "bbox": bbox}
    a.handle("voice", noise)
    assert requests(shared) == []


def test_the_vlm_is_told_which_side_of_the_patient_the_camera_sees(presets):
    a, _shared, backend = make(
        presets, [{"reply": "どちらですか。", "decision": "ask", "preset_id": None, "target": ""}]
    )
    seen = []
    decide = backend.decide
    backend.decide = lambda instructions, *rest: seen.append(instructions) or decide(instructions, *rest)
    a.handle("text", "左肩")
    assert presets.view == "back" and "背中側" in seen[0] and "人の左は画像の左側" in seen[0]
    assert "{sides}" not in seen[0]


def test_a_grid_shows_cells_but_the_policy_gets_the_trained_size_bbox(tmp_path):
    path = tmp_path / "grid.yaml"
    path.write_text(
        "image_size: [1280, 720]\ncell_px: [75, 75]\npresets:\n"
        "  - {id: 1, name: a, bbox: [408, 340, 526, 495]}\n  - {id: 2, name: b, bbox: [483, 340, 601, 495]}\n",
        encoding="utf-8",
    )
    grid = Presets.load(path)
    assert grid.cell == (75.0, 75.0)
    np.testing.assert_allclose(grid.shown(grid.items[0]), [429.5, 380, 504.5, 455])
    status = grid.as_status()[0]
    np.testing.assert_allclose(grid.pixels(status["bbox"]), [408, 340, 526, 495])
    np.testing.assert_allclose(grid.pixels(status["shown"]), [429.5, 380, 504.5, 455])
    _ok, jpeg = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))
    assert cv2.imdecode(np.frombuffer(annotate(jpeg.tobytes(), grid), np.uint8), 1).sum() > 0

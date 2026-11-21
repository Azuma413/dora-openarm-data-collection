"""Voice dialogue: decide where the patch goes with a VLM, then take corrections.

The browser does the listening: it detects speech locally (so a voice stops the
robot at once, before anything reaches OpenAI), records the utterance and posts
it here. This thread transcribes it, asks the VLM what to do, speaks the reply
(the browser plays it) and asks the robot loop to act through the same request
queue the UI buttons use.

Two phases:

* ``locate`` -- the user says where ("左肩に貼って"). The VLM sees head_left with
  the presets drawn as numbered boxes and asks follow-up questions until the
  place is unique, then picks the closest preset. The bbox is set, the reply is
  spoken, and the policy starts once the browser has played it.
* ``execute`` -- the policy has a target. Speech stops the robot (the browser
  does that on the first sound). The utterance then becomes: adjust (move the
  bbox by whole steps, clamped to the presets' range), resume, stop, relocate (a
  new preset) or ask. Adjust, resume and relocate restart the policy after the
  reply has been spoken.
"""

from __future__ import annotations

import base64
import collections
import io
import json
import queue
import threading
import time
import traceback
import wave
from typing import Any, Protocol

import numpy as np

from dora_openarm_policy.presets import Presets, annotate

LOCATE = "locate"
EXECUTE = "execute"

# Whisper-style transcripts of silence or noise: treated as nothing said.
NOISE_TRANSCRIPTS = frozenset(
    {"", "ご視聴ありがとうございました", "ありがとうございました", "おやすみなさい", "あ", "えー", "えっと"}
)
# What a voice stop answers when nothing usable was said.
SAY_STOPPED = "止まりました。指示をどうぞ。"

LOCATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "decision", "preset_id", "target"],
    "properties": {
        "reply": {"type": "string"},
        "decision": {"type": "string", "enum": ["ask", "decided", "other"]},
        "preset_id": {"type": ["integer", "null"]},
        "target": {"type": "string"},
    },
}

CORRECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "action", "dx_steps", "dy_steps", "preset_id"],
    "properties": {
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": ["adjust", "resume", "stop", "relocate", "ask"]},
        "dx_steps": {"type": "integer"},
        "dy_steps": {"type": "integer"},
        "preset_id": {"type": ["integer", "null"]},
    },
}

LOCATE_INSTRUCTIONS = """あなたは、人に湿布を貼るロボットの音声対話係です。
- 画像はロボットの頭のカメラ（ロボットの前方を見ている）の画像です。湿布を貼られる人（マネキンのこともあります）が写っています。
- 画像には、ロボットが湿布を貼れる位置の候補（プリセット）が、番号付きのオレンジ色の枠（格子のときはマスの中央に番号）で描かれています。
- ユーザーの発話は音声認識の書き起こしで、誤認識を含むことがあります。

目的: 湿布を貼る位置を一意に決め、もっとも合うプリセットを 1 つ選ぶこと。

手順:
1. 位置が一意に決まらないうち（例:「左肩」だけでは肩の上側か下側か分からない）は decision="ask" とし、位置を絞り込む短い質問を 1 つだけ reply に書く。選択肢を 2〜3 個挙げると答えやすい。
2. 位置が一意に決まったら decision="decided" とする。preset_id にその位置にもっとも近いプリセットの番号、target に決まった位置の短い説明を書き、reply で位置を復唱して「貼ります」と伝える。
3. 指示された位置がどのプリセットからも明らかに離れているときは decision="other" とし、そこには貼れないことと、貼れるおおよその範囲を reply で伝える。
4. 湿布と関係のない発話には decision="other" で短く答える。

注意:
- 左右は、貼られる人自身の左右です。{sides}
- decided 以外では preset_id は null。
- reply は音声で読み上げます。日本語で 1〜2 文、丁寧に短く。番号・記号・英語は使わないでください。"""

CORRECT_INSTRUCTIONS = """あなたは、人に湿布を貼るロボットの音声対話係です。ロボットは目標位置（画像の緑の枠）に湿布を貼る動作をしていましたが、ユーザーが声をかけたので一時停止しています。
画像はロボットの頭のカメラ（ロボットの前方を見ている）の今の画像です。緑の枠が今の目標位置、番号付きのオレンジの枠が貼れる位置の候補です。
ユーザーの発話（音声認識の書き起こし。誤認識を含むことがある）から、次の行動を 1 つ選んでください。

- adjust: 位置の修正（「もっと上」「少し右」など）。dx_steps と dy_steps に、画像上での移動量をステップ数で書く。画像の右が +x、下が +y。1 ステップは約 {step} ピクセル（画像は 1280×720）。目安は「少し」=1、指定なし=2、「もっと大きく」「かなり」=3〜4。
- resume: 位置はそのままで再開する（「大丈夫」「続けて」など）。
- stop: 止まったままにする（「止まって」「やめて」、痛みや危険を感じさせる発話など）。
- relocate: まったく別の場所を指示された。preset_id にもっとも合うプリセットの番号を書く。
- ask: 意図が分からない。reply で短く聞き返す。

注意:
- 上下は体の上下（頭の方が上）。左右は貼られる人自身の左右です。{sides}「画面の右」のように画像の向きで言われたら、画像の向きに従う。体の方向を画像上の dx_steps / dy_steps に直してください。
- adjust 以外では dx_steps と dy_steps は 0。relocate 以外では preset_id は null。
- reply は音声で読み上げます。日本語で 1 文、短く（例:「少し上にずらします」）。
- 迷ったら安全側（stop か ask）を選んでください。"""


# An utterance whose peak is below this is noise the browser took for a voice
# (the motors starting up), not speech: dropped before transcription. Speech on
# the lab's microphone peaked at -34 to -40 dBFS; the false ones below -49.
MIN_SPEECH_PEAK_DBFS = -46.0
# Utterances are brought to this peak before transcription (-3 dBFS)...
NORMALIZE_PEAK = 0.7
# ...by at most this gain (+46 dB), so a near-silent recording stays quiet noise.
# Only utterances the browser heard as speech get here, and the lab's microphone
# records a voice at -40 to -55 dBFS.
NORMALIZE_MAX_GAIN = 200.0


def wav_peak_dbfs(wav: bytes) -> float | None:
    """The 99.9th-percentile sample level of a 16-bit WAV in dBFS (None if unreadable)."""
    try:
        with wave.open(io.BytesIO(wav)) as w:
            if w.getsampwidth() != 2:
                return None
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if not frames:
        return None
    x = np.frombuffer(frames, np.int16).astype(np.float64) / 32768.0
    return float(20 * np.log10(np.percentile(np.abs(x), 99.9) + 1e-12))


def normalize_wav(wav: bytes) -> tuple[bytes, float]:
    """The utterance at a fixed loudness, and the gain applied (1.0 = unchanged).

    The microphone may record speech 30 dB below full scale (the lab's TAMAGO-03
    does, at -50 dBFS). The peak is taken as the 99.9th percentile of the samples,
    so a single click does not set it. Anything but 16-bit PCM is returned as is.
    """
    try:
        with wave.open(io.BytesIO(wav)) as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
    except (wave.Error, EOFError):
        return wav, 1.0
    if params.sampwidth != 2 or not frames:
        return wav, 1.0
    x = np.frombuffer(frames, np.int16).astype(np.float64) / 32768.0
    peak = float(np.percentile(np.abs(x), 99.9))
    if peak <= 0.0:
        return wav, 1.0
    gain = min(NORMALIZE_PEAK / peak, NORMALIZE_MAX_GAIN)
    y = np.clip(np.round(x * gain * 32767.0), -32768, 32767).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(y.tobytes())
    return out.getvalue(), gain


class Backend(Protocol):
    def transcribe(self, wav: bytes) -> str: ...
    def speak(self, text: str) -> bytes: ...
    def decide(self, instructions: str, prompt: str, image_jpeg: bytes, schema: dict, name: str) -> dict: ...


class OpenAIBackend:
    """Speech to text, the VLM and text to speech on the OpenAI API."""

    def __init__(self, api_key: str, vlm_model: str, stt_model: str, tts_model: str, tts_voice: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=30.0)
        self.vlm_model, self.stt_model = vlm_model, stt_model
        self.tts_model, self.tts_voice = tts_model, tts_voice

    def transcribe(self, wav: bytes) -> str:
        result = self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=("speech.wav", io.BytesIO(wav), "audio/wav"),
            language="ja",
            # No vocabulary prompt: given motor noise, the model transcribed the
            # prompt's words ("右肩", "続けて") as if they had been said.
        )
        return str(getattr(result, "text", "") or "").strip()

    def speak(self, text: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=text,
            instructions="日本語で、落ち着いた丁寧な口調で、はっきり話してください。",
            response_format="mp3",
        )
        return response.read()

    def decide(self, instructions: str, prompt: str, image_jpeg: bytes, schema: dict, name: str) -> dict:
        image = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        response = self.client.responses.create(
            model=self.vlm_model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image},
                    ],
                }
            ],
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        )
        return json.loads(response.output_text)


class Assistant:
    def __init__(
        self,
        shared,
        presets: Presets,
        backend: Backend | None,
        log,
        *,
        speech_timeout_s: float = 20.0,
        ready_timeout_s: float = 120.0,
        setup_error: str | None = None,
    ) -> None:
        self.shared = shared
        self.presets = presets
        self.backend = backend
        self.log = log
        self.speech_timeout_s = speech_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.phase = LOCATE
        self.history: list[dict[str, str]] = []
        self.target = ""
        self.preset_id: int | None = None
        self.busy: str | None = None
        self.error: str | None = setup_error
        self.speech: dict[str, Any] | None = None
        self._speech_seq = 0
        # (speech id, what to do once it has been played, deadline)
        self._after_speech: tuple[int, str, float] | None = None
        self._start_by: float | None = None  # start the policy once it can start, until then
        self._told_waiting = False
        self.done = threading.Event()
        self._publish()

    # -- the thread ---------------------------------------------------------

    def run(self) -> None:
        while not self.done.is_set():
            try:
                kind, payload = self.shared.assistant_inbox.get(timeout=0.1)
            except queue.Empty:
                kind = None
            try:
                if kind is not None:
                    self.handle(kind, payload)
                self.poll(time.monotonic())
            except Exception as error:  # noqa: BLE001 - reported in the UI, the thread carries on
                traceback.print_exc()
                self.error = f"{type(error).__name__}: {error}"
                self.busy = None
                self.log(f"assistant: {self.error}")
                self._publish()

    def handle(self, kind: str, payload: Any) -> None:
        if kind == "voice":
            self._on_voice(payload)
        elif kind == "text":
            self._on_utterance(str(payload), voice=False)
        elif kind == "speech_done":
            self._on_speech_done(int(payload))
        elif kind == "reset":
            self.reset()

    def poll(self, now: float) -> None:
        if self._after_speech is not None and now > self._after_speech[2]:
            # The browser never said it finished (closed, muted): go on anyway.
            _, then, _ = self._after_speech
            self._after_speech = None
            self._do(then)
        if self._start_by is not None:
            status = self._status()
            if status.get("mode") == "running":
                self._start_by = None
            elif status.get("can_start") is None and status.get("mode") is not None:
                self._start_by = None
                self._request("action", "start")
                self.log("assistant: the policy can start now; starting it")
            elif now > self._start_by:
                self._start_by = None
                self.log(f"assistant: gave up waiting to start ({status.get('can_start')})")
            elif not self._told_waiting and "intro" in str(status.get("can_start")):
                self._told_waiting = True
                self._say("湿布を受け取ったら、すぐに始めます。")

    def reset(self) -> None:
        self.phase = LOCATE
        self.history.clear()
        self.target = ""
        self.preset_id = None
        self._after_speech = None
        self._start_by = None
        self.error = None
        self.log("assistant: dialogue reset")
        self._publish()

    # -- utterances ---------------------------------------------------------

    def _on_voice(self, wav: bytes) -> None:
        if self.backend is None:
            self.log(f"assistant: voice ignored ({self.error})")
            return
        peak = wav_peak_dbfs(wav)
        if peak is not None and peak < MIN_SPEECH_PEAK_DBFS:
            status = self._status()
            if status.get("mode") == "stopped" and status.get("reason") == "voice" and self.phase == EXECUTE:
                # The browser stopped the robot for a sound that was not speech:
                # nobody said anything, so carry on as before.
                self.log(f"assistant: noise stopped the robot (peak {peak:.0f} dBFS); resuming")
                self._request("action", "start")
            else:
                self.log(f"assistant: utterance dropped as noise (peak {peak:.0f} dBFS)")
            return
        self._set_busy("transcribing")
        wav, gain = normalize_wav(wav)
        if abs(gain - 1.0) > 0.05:
            self.log(f"assistant: utterance gain {20 * np.log10(gain):+.0f} dB")
        text = self.backend.transcribe(wav)
        self._on_utterance(text, voice=True)

    def _on_utterance(self, text: str, voice: bool) -> None:
        text = text.strip()
        if text.strip("。、.!！?？ ") in NOISE_TRANSCRIPTS:
            self._set_busy(None)
            self.log(f"assistant: nothing usable heard ({text!r})")
            if voice and self._status().get("mode") == "stopped" and self.phase == EXECUTE:
                self._say(SAY_STOPPED)
            return
        if self.backend is None:
            self.log(f"assistant: {self.error}")
            return
        self.history.append({"role": "user", "text": text})
        self.log(f"assistant: heard {text!r}")
        self._set_busy("thinking")
        if self.phase == LOCATE:
            self._locate(text)
        else:
            self._correct(text)

    def _image(self, live: bool) -> bytes | None:
        with self.shared.lock:
            if live:
                return self.shared.head_jpeg or self.shared.snapshot_jpeg
            return self.shared.snapshot_jpeg or self.shared.head_jpeg

    def _preset_lines(self) -> str:
        w, h = self.presets.image_size
        lines = []
        for p in self.presets.items:
            cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
            lines.append(f"- 番号 {p.id}: {p.name}（枠の中心 x={cx:.0f}, y={cy:.0f}、画像 {w}×{h}）")
        return "\n".join(lines)

    def _transcript(self) -> str:
        turns = self.history[-12:-1]
        if not turns:
            return "（なし）"
        return "\n".join(("ユーザー: " if t["role"] == "user" else "ロボット: ") + t["text"] for t in turns)

    def _locate(self, text: str) -> None:
        jpeg = self._image(live=False)
        if jpeg is None:
            self._reply("カメラの画像がまだありません。少し待ってからもう一度お願いします。")
            return
        prompt = (
            f"プリセット:\n{self._preset_lines()}\n\nこれまでの会話:\n{self._transcript()}\n\n"
            f"ユーザーの最新の発話:「{text}」"
        )
        instructions = LOCATE_INSTRUCTIONS.replace("{sides}", self.presets.sides_text())
        result = self.backend.decide(
            instructions, prompt, annotate(jpeg, self.presets), LOCATE_SCHEMA, "locate"
        )
        reply = str(result.get("reply") or "").strip()
        preset = self.presets.get(result.get("preset_id")) if result.get("preset_id") is not None else None
        if result.get("decision") == "decided" and preset is not None:
            self.target = str(result.get("target") or preset.name)
            self.preset_id = preset.id
            self.phase = EXECUTE
            self._request("bbox", self.presets.normalized(preset.bbox))
            self.log(f"assistant: target {self.target!r} -> preset {preset.id} ({preset.name})")
            self._reply(reply or f"{self.target}に貼ります。", then="start")
        else:
            if result.get("decision") == "decided":
                self.log(f"assistant: the VLM picked an unknown preset {result.get('preset_id')!r}")
            self._reply(reply or "どこに貼りましょうか。")

    def _correct(self, text: str) -> None:
        status = self._status()
        bbox = status.get("bbox")
        jpeg = self._image(live=True)
        if bbox is None or jpeg is None:
            self.phase = LOCATE
            self._locate(text)
            return
        current = self.presets.pixels(bbox)
        prompt = (
            f"プリセット:\n{self._preset_lines()}\n\n今の目標位置（緑の枠）: 中心 x={(current[0] + current[2]) / 2:.0f}, "
            f"y={(current[1] + current[3]) / 2:.0f}（{self.target or '指定済みの位置'}）\n\n"
            f"これまでの会話:\n{self._transcript()}\n\nユーザーの最新の発話:「{text}」"
        )
        instructions = CORRECT_INSTRUCTIONS.replace("{step}", f"{self.presets.step_px:.0f}").replace(
            "{sides}", self.presets.sides_text()
        )
        result = self.backend.decide(
            instructions, prompt, annotate(jpeg, self.presets, current), CORRECT_SCHEMA, "correct"
        )
        action = result.get("action")
        reply = str(result.get("reply") or "").strip()
        if action == "adjust":
            dx, dy = int(result.get("dx_steps") or 0), int(result.get("dy_steps") or 0)
            moved, clamped = self.presets.shift(current, dx, dy)
            self._request("bbox", self.presets.normalized(moved))
            self.log(
                f"assistant: adjust {dx:+d}, {dy:+d} steps{' (clamped to the presets range)' if clamped else ''}"
            )
            if clamped:
                reply = (reply + " " if reply else "") + "これ以上は動かせないので、届く範囲で貼ります。"
            self._reply(reply or "位置を直します。", then="start")
        elif action == "relocate" and self.presets.get(result.get("preset_id")) is not None:
            preset = self.presets.get(result.get("preset_id"))
            self.preset_id = preset.id
            self.target = preset.name
            self._request("bbox", self.presets.normalized(preset.bbox))
            self.log(f"assistant: relocate -> preset {preset.id}")
            self._reply(reply or "場所を変えます。", then="start")
        elif action == "resume":
            self._reply(reply or "続けます。", then="start")
        else:
            self._reply(reply or SAY_STOPPED)

    # -- speaking and acting ------------------------------------------------

    def _reply(self, text: str, then: str | None = None) -> None:
        self.history.append({"role": "assistant", "text": text})
        self._say(text, then)

    def _say(self, text: str, then: str | None = None) -> None:
        self._set_busy("speaking")
        audio = None
        try:
            audio = self.backend.speak(text) if self.backend is not None else None
        except Exception as error:  # noqa: BLE001 - shown; the action still happens
            self.error = f"TTS: {type(error).__name__}: {error}"
            self.log(f"assistant: {self.error}")
        self._speech_seq += 1
        speech_id = self._speech_seq
        if audio:
            with self.shared.lock:
                self.shared.tts[speech_id] = audio
                while len(self.shared.tts) > 20:
                    self.shared.tts.popitem(last=False)
        self.speech = {"id": speech_id, "text": text, "audio": bool(audio)}
        self.log(f"assistant says: {text}")
        self.busy = None
        if then is not None:
            if audio:
                self._after_speech = (speech_id, then, time.monotonic() + self.speech_timeout_s)
            else:
                self._do(then)
        self._publish()

    def _on_speech_done(self, speech_id: int) -> None:
        if self._after_speech is not None and self._after_speech[0] == speech_id:
            _, then, _ = self._after_speech
            self._after_speech = None
            self._do(then)

    def _do(self, then: str) -> None:
        if then == "start":
            self._start_by = time.monotonic() + self.ready_timeout_s
            self._told_waiting = False
            self.poll(time.monotonic())

    def _request(self, kind: str, payload: Any) -> None:
        self.shared.requests.put((kind, payload))

    def _status(self) -> dict:
        with self.shared.lock:
            return dict(self.shared.status)

    def _set_busy(self, busy: str | None) -> None:
        self.busy = busy
        self._publish()

    def _publish(self) -> None:
        status = {
            "enabled": self.backend is not None,
            "phase": self.phase,
            "busy": self.busy,
            "error": self.error,
            "target": self.target,
            "preset_id": self.preset_id,
            "speech": self.speech,
            "history": self.history[-20:],
            "waiting_to_start": self._start_by is not None,
        }
        with self.shared.lock:
            self.shared.assistant_status = status


def new_tts_store() -> collections.OrderedDict:
    return collections.OrderedDict()

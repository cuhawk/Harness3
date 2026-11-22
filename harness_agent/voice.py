"""Voice I/O for the REPL: speak replies aloud, and take turns from the microphone.

Everything runs on the CPU against local model files - the RTX 5080 is never touched, and no
audio or text leaves the machine. The constants here are not guesses; they came out of a
benchmark on this box (see Claude Folder\\voice-lab), and several are the opposite of the
obvious choice:

  * cpu_threads=24. ctranslate2's default pick leaves most of a 12-core/24-thread HX 370
    idle - 1.95s per clip default versus 1.22s at 24, with identical accuracy.
  * small.en, not something smaller. base.en actually scores WORSE than tiny.en (19.1% vs
    13.7% WER) and distil-small.en is worse than both with no speed gain. small.en lands at
    4.4%, and it is the only one that survives security jargon: the smaller models turn
    "unsafe eval" into "unsafe a valve" and "Mistral Large" into "Mr. Large".
  * Kokoro fp32, not int8. The quantised build is 4x SLOWER on x86 (2.86s vs 0.70s to first
    clause) because it misses the optimised kernels.
  * HF_HUB_OFFLINE=1. huggingface_hub re-checks the repo on every model load even when the
    weights are fully cached, which is a real network call in a stack whose whole point is
    that it is local.

This module imports cleanly with none of the optional dependencies installed - everything
audio is imported lazily inside functions, and available() reports what is missing - so a
harness without them behaves exactly as it did before.
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import core

# Must be set before faster_whisper imports huggingface_hub; harmless if it already is.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

RATE = 16000
CPU_THREADS = 24
# Whisper decodes a silent buffer as the literal string "You" (YouTube caption filler) rather
# than returning nothing, so a muted mic is indistinguishable from a broken model unless the
# level is checked first. Belt and braces: vad_filter below returns '' for the same case.
SILENCE_FLOOR = 0.005

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/][\w.\-\\/]{6,}")
_MD = re.compile(r"[*_#`>|]+")

_stt_model: Any = None
_speaker: "Speaker | None" = None
# A mood held across replies until cleared, set by /voice mood. Nothing sets it automatically:
# detection reads the REPLY, so it can tell a finding is alarming but never that a tone was
# requested. This is the knob for that.
_pinned_mood: str | None = None
# Strips a model-chosen [mood: x] off the front of the reply being streamed. Rebuilt per
# message by begin_stream, None when VOICE_MOOD_TAGS is off.
_tagger: "MoodTag | None" = None
_cues: dict[str, dict] | None = None
_lock = threading.Lock()


def phrasebank_dir() -> Path:
    override = os.environ.get("VOICE_PHRASEBANK")
    if override is not None:
        return Path(override)
    return Path.home() / ".harness" / "voice" / "phrasebank"


def cues() -> dict[str, dict]:
    """Short emotionally-coloured clips pre-rendered by Chatterbox (see voice-lab/
    cbx_phrasebank.py), loaded once and cached.

    Kokoro cannot do emotion at all - measured, 13 Hz of pitch movement across every mood
    treatment against the 50-100 Hz+ of real emotional speech. Chatterbox can (102 Hz) but
    runs at 0.24x realtime on CPU, and the GPU is entirely occupied by Qwen. Pre-rendering
    the dozen short utterances that actually carry affect is the only way to get both: the
    cue is emotional and free at play time, the body stays Kokoro and stays fast.
    """
    global _cues
    if _cues is None:
        index = phrasebank_dir() / "index.json"
        try:
            _cues = json.loads(index.read_text(encoding="utf-8"))
        except Exception:
            _cues = {}
    return _cues


# ---------------------------------------------------------------------------
# Availability


def missing_deps() -> list[str]:
    """Which optional packages are absent. Checked by importlib rather than a try/import so
    asking the question never costs the (multi-second) import itself."""
    import importlib.util

    wanted = {"faster_whisper": "faster-whisper", "kokoro_onnx": "kokoro-onnx",
              "sounddevice": "sounddevice"}
    return [pkg for mod, pkg in wanted.items() if importlib.util.find_spec(mod) is None]


def tts_model_path() -> Path:
    override = os.environ.get("VOICE_TTS_MODEL")
    if override is not None:
        return Path(override)
    return Path.home() / ".harness" / "voice" / "models" / "kokoro-v1.0.onnx"


def tts_voices_path() -> Path:
    return tts_model_path().with_name("voices-v1.0.bin")


def available() -> tuple[bool, str]:
    """(usable, reason). Reason is only meaningful when usable is False."""
    if missing := missing_deps():
        return False, f"missing packages: {', '.join(missing)} (pip install {' '.join(missing)})"
    if not tts_model_path().exists():
        return False, f"no Kokoro model at {tts_model_path()} (set VOICE_TTS_MODEL)"
    return True, ""


# ---------------------------------------------------------------------------
# What actually gets read aloud


def speakable(text: str) -> str:
    """Strip a written reply down to something worth listening to.

    A harness reply is mostly paths, JSON and fenced code; read verbatim it is unlistenable,
    and a synthesiser spelling out a 60-character Windows path is worse than useless. Lossy
    on purpose - the terminal still shows the real text, this is only what reaches the
    speakers.
    """

    def fence(m: re.Match) -> str:
        n = max(len(m.group(0).strip().splitlines()) - 2, 1)
        return f" code block, {n} lines. "

    # Safety net for a model that puts its mood tag somewhere the streaming stripper cannot
    # see it. Better dropped silently than read out as "bracket mood colon angry".
    text = _MOOD_TAG_ANY.sub(" ", text)
    text = _CODE_FENCE.sub(fence, text)
    text = _URL.sub(" a link ", text)
    text = _PATH.sub(lambda m: " " + Path(m.group(0)).name + " ", text)
    text = _MD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Output


# Tolerant on purpose. Asked for "[mood: none]", Qwen answers "[None]" often enough that a
# strict matcher leaves the tag in the reply, where it gets read aloud. The bare form is only
# accepted when the word is actually a mood (or "none"), so a reply that genuinely opens with
# "[see below]" is left alone.
_MOOD_TAG = re.compile(r"^\s*\[\s*(mood\s*[:\-]\s*)?([a-z_]+)\s*\]\s*", re.I)
_MOOD_TAG_ANY = re.compile(r"\[\s*(?:mood\s*[:\-]\s*[a-z_]+|none)\s*\]", re.I)
_TAG_WINDOW = 40   # a tag is ~14 chars; decide well before a clause could have been spoken


class MoodTag:
    """Pull a leading [mood: x] off a reply as it streams.

    Streaming is what makes this awkward: the tag arrives a few tokens at a time, so the match
    has to wait for enough characters - but not so long that speech is delayed. Text is held
    only until the tag resolves or the window fills, which is inside the first clause.
    """

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.mood: str | None = None
        self._buf = ""
        self._done = False

    def feed(self, chunk: str) -> str:
        if self._done:
            return chunk
        self._buf += chunk
        if m := _MOOD_TAG.match(self._buf):
            explicit, name = bool(m.group(1)), m.group(2).lower()
            if explicit or name in self.known or name == "none":
                # An invented mood means neutral, not a crash.
                self.mood = name if name in self.known else None
                self._done = True
                return self._buf[m.end():]
            # A bare bracket that is not a mood is content. Leave it exactly as written.
            self._done = True
            return self._buf
        if len(self._buf) >= _TAG_WINDOW or "\n" in self._buf:
            self._done = True
            return self._buf
        return ""

    def flush(self) -> str:
        if self._done:
            return ""
        self._done = True
        return self._buf


def tone_for(mood: str | None) -> tuple[float, float]:
    """(exaggeration, cfg_weight) for a mood, from moods.json - the same numbers the rendered
    cue bank used, so a mood sounds the same live as it does canned."""
    spec = load_moods().get(mood or "", {})
    return (float(spec.get("exaggeration", 0.5)), float(spec.get("cfg_weight", 0.5)))


class KokoroEngine:
    """Fast, tiny, tone-deaf: 4x realtime on CPU, no VRAM, and no emotion axis whatsoever."""

    name = "kokoro"
    emotional = False

    def __init__(self, voice: str) -> None:
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(str(tts_model_path()), str(tts_voices_path()))
        self.voice = voice

    def synth(self, text: str, mood: str | None = None) -> Any:
        return self.kokoro.create(text, voice=self.voice, lang="en-us")


class ChatterboxEngine:
    """Client for cbx_server.py, holding Chatterbox resident on the GPU.

    Measured on cuda 2026-08-22: 3674 MiB peak, 1.5-2.05x realtime, 6s to load. On CPU it is
    0.24x realtime, which is why this was a pre-rendered cue bank until the VRAM was found.
    """

    name = "chatterbox"
    emotional = True

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        import requests

        self.url = url.rstrip("/")
        self.timeout = timeout
        r = requests.get(f"{self.url}/health", timeout=4)
        r.raise_for_status()
        self.health = r.json()
        # Chatterbox ships ONE voice; the rest are clones of reference wavs the server found
        # in voice-lab\audio\voices. Measured spread across the Kokoro-derived set: 94 Hz
        # (am_onyx) to 197 Hz (af_heart) - as wide as the whole emotion range.
        self.available: list[str] = self.health.get("voices") or ["default"]
        want = core.VOICE_TTS_SPEAKER
        self.voice = want if want in self.available else "default"
        if want != self.voice:
            print(f"[voice] no cloned voice {want!r} on the tts server; using default. "
                  f"have: {', '.join(self.available)}", file=sys.stderr)

    def synth(self, text: str, mood: str | None = None) -> Any:
        import io

        import requests
        import soundfile as sf

        exag, cfg = tone_for(mood)
        r = requests.post(f"{self.url}/tts", timeout=self.timeout,
                          json={"text": text, "exaggeration": exag, "cfg_weight": cfg,
                                "voice": self.voice or "default"})
        r.raise_for_status()
        return sf.read(io.BytesIO(r.content), dtype="float32")


class Speaker:
    """Synthesis and playback on separate threads.

    Two threads rather than one because Kokoro runs at roughly 4x realtime: synthesising
    clause N+1 while clause N is still playing hides the cost entirely, whereas doing both on
    one thread leaves an audible ~0.8s gap between every sentence.
    """

    def __init__(self, engine, voice: str) -> None:
        self.engine = engine
        self.voice = voice
        # Items are tagged ("text", str) or ("cue", Path). A pre-rendered cue could be pushed
        # straight to audio_q since it is already (samples, rate), but that would let it
        # overtake a sentence still being synthesised - so cues ride the same queue and the
        # synth thread passes them through, which preserves order for free.
        self.text_q: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self.audio_q: queue.Queue[Any] = queue.Queue(maxsize=4)
        self.interrupted = threading.Event()
        self._buf = ""
        self._cued = False
        # Mood for the message being spoken. It rides with every sentence of that message so
        # an emotional engine colours the whole reply; re-detecting per sentence would make
        # the tone lurch mid-answer. None means "not decided yet"; begin(mood) pins it.
        self._mood: str | None = None
        # Items in flight: queued, dequeued by _synth and mid-synthesis, or waiting in
        # audio_q. Counting them is the only way drain() can know a reply finished - the
        # "both queues empty" test reports done for an item _synth has taken but not yet
        # handed on, so a following interrupt() silences audio that was never heard.
        self._pending = 0
        self._cv = threading.Condition()
        threading.Thread(target=self._synth, daemon=True, name="voice-synth").start()
        threading.Thread(target=self._play, daemon=True, name="voice-play").start()

    def _track(self, delta: int) -> None:
        with self._cv:
            self._pending = max(0, self._pending + delta)
            self._cv.notify_all()

    def _submit(self, item: tuple[str, Any]) -> None:
        self._track(+1)
        self.text_q.put(item)

    def _synth(self) -> None:
        while True:
            item = self.text_q.get()
            if item is None:
                continue
            if self.interrupted.is_set():
                self._track(-1)
                continue
            kind, payload = item
            try:
                if kind == "cue":
                    import soundfile as sf

                    samples, rate = sf.read(str(payload), dtype="float32")
                    self.audio_q.put((samples, rate))
                else:
                    text, mood = payload
                    self.audio_q.put(self.engine.synth(text, mood))
            except Exception as exc:  # a TTS failure must never kill the turn
                self._track(-1)
                print(f"[voice] synthesis failed: {exc}", file=sys.stderr)

    def _play(self) -> None:
        import sounddevice as sd

        while True:
            item = self.audio_q.get()
            if item is None:
                continue
            if self.interrupted.is_set():
                self._track(-1)
                continue
            samples, rate = item
            try:
                sd.play(samples, rate)
                while sd.get_stream().active:
                    if self.interrupted.is_set():
                        sd.stop()
                        break
                    time.sleep(0.03)
            except Exception as exc:
                print(f"[voice] playback failed: {exc}", file=sys.stderr)
            finally:
                self._track(-1)

    # -- whole messages ---------------------------------------------------

    def _enqueue_cue(self, mood: str) -> None:
        entry = cues().get(mood)
        if not entry:
            return
        variants = entry.get("variants") or ([entry] if entry.get("file") else [])
        if variants:
            self._submit(("cue", phrasebank_dir() / random.choice(variants)["file"]))

    def begin(self, mood: str | None = None) -> None:
        """Reset per-message state at the start of a streamed reply.

        A mood passed here is PINNED - detection is skipped for this message. That is how
        /voice mood beats the trigger regexes, which only ever see the reply text and so can
        never tell that a tone was asked for rather than warranted by a finding.
        """
        self._buf = ""
        self._cued = False
        self._mood = mood

    def say(self, text: str, mood: str | None = None) -> None:
        cleaned = speakable(text)
        if cleaned:
            self.interrupted.clear()
            self._mood = mood
            self._submit(("text", (cleaned, mood)))

    def set_mood(self, mood: str | None) -> None:
        """Pin the mood for the message already in progress. The model's own tag arrives after
        begin() but before the first complete sentence, which is exactly this window."""
        if mood is not None:
            self._mood = mood

    # -- streamed messages ------------------------------------------------

    def feed(self, chunk: str) -> None:
        """Accumulate streamed tokens, flushing each complete sentence to the synthesiser as
        soon as it lands. This is what lets speech start about a second into a reply instead
        of after the model has finished writing it."""
        self.interrupted.clear()
        self._buf += chunk
        parts = _SENT_END.split(self._buf)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if cleaned := speakable(sentence):
                    # The cue has to land BEFORE the first sentence, and the first sentence is
                    # the only text available at that point - which is fine, because a reply
                    # states its verdict up front.
                    if not self._cued:
                        self._cued = True
                        if self._mood is None:   # None means nothing pinned - go detect
                            self._mood = detect_mood(cleaned)
                        # A cue is a canned opener in front of a flat body - the workaround
                        # for Kokoro having no tone. An engine that can colour the body makes
                        # it redundant, and it costs a second to say what the reply says.
                        if core.VOICE_CUES and self._mood and not self.engine.emotional:
                            self._enqueue_cue(self._mood)
                    self._submit(("text", (cleaned, self._mood)))
            self._buf = parts[-1]
        elif len(self._buf) > 220:
            # Some models emit a whole paragraph before any terminal punctuation; waiting for
            # one would defeat the point of streaming.
            if cleaned := speakable(self._buf):
                self._submit(("text", (cleaned, self._mood)))
            self._buf = ""

    def flush(self) -> None:
        if cleaned := speakable(self._buf):
            self._submit(("text", (cleaned, self._mood)))
        self._buf = ""

    def stop(self) -> None:
        """Barge-in: drop everything queued and silence playback immediately."""
        self.interrupted.set()
        self._buf = ""
        with self._cv:
            self._pending = 0
            self._cv.notify_all()
        for q in (self.text_q, self.audio_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


def _get_speaker() -> "Speaker | None":
    global _speaker
    with _lock:
        if _speaker is None:
            ok, reason = available()
            if not ok:
                print(f"[voice] output disabled: {reason}", file=sys.stderr)
                core.VOICE_OUT = False
                return None
            _speaker = Speaker(_build_engine(), core.VOICE_TTS_VOICE)
        return _speaker


def _build_engine() -> Any:
    """Chatterbox if asked for and reachable, otherwise Kokoro.

    Falling back rather than raising is deliberate: a TTS server that is down should cost
    tone, not speech. It says so on stderr rather than failing silently, because a suddenly
    flat voice is exactly what gets reported as "the moods stopped working".
    """
    if core.VOICE_TTS_ENGINE == "chatterbox":
        try:
            eng = ChatterboxEngine(core.VOICE_CBX_URL)
            if eng.health.get("device") != "cuda":
                print(f"[voice] chatterbox is on {eng.health.get('device')}, not cuda - "
                      "expect ~0.24x realtime", file=sys.stderr)
            return eng
        except Exception as exc:
            print(f"[voice] chatterbox at {core.VOICE_CBX_URL} is not answering ({exc}); "
                  "using kokoro, which has no emotion axis. Start it with "
                  "start-voice-gpu.ps1", file=sys.stderr)
            core.VOICE_TTS_ENGINE = "kokoro"
    return KokoroEngine(core.VOICE_TTS_VOICE)


# Moods live in moods.json next to the phrasebank - phrases, intensity and trigger regex all
# in one file - so tuning never means editing Python. cbx_phrasebank.py renders from the same
# file. See its header for what each knob does.
_moods_cache: dict[str, dict] | None = None
_triggers_cache: list[tuple[str, Any]] | None = None


def moods_file() -> Path:
    return Path(os.environ.get("VOICE_MOODS", str(phrasebank_dir().parent.parent / "moods.json")))


def load_moods() -> dict[str, dict]:
    """moods.json, or {} if missing/unreadable - a broken bank must never break speech."""
    global _moods_cache
    if _moods_cache is None:
        try:
            data = json.loads(moods_file().read_text(encoding="utf-8"))
            _moods_cache = {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            _moods_cache = {}
    return _moods_cache


def triggers() -> list[tuple[str, Any]]:
    """Compiled (mood, pattern) pairs in file order - first match wins, so moods.json lists
    the alarming moods first. A null trigger means manual-only."""
    global _triggers_cache
    if _triggers_cache is None:
        out = []
        for mood, spec in load_moods().items():
            if pat := spec.get("trigger"):
                try:
                    out.append((mood, re.compile(pat, re.I)))
                except re.error as exc:
                    print(f"[voice] bad trigger for {mood!r}: {exc}", file=sys.stderr)
        _triggers_cache = out
    return _triggers_cache


def reload_moods() -> int:
    """Drop caches so an edit to moods.json applies without restarting. The audio itself only
    changes after a re-render."""
    global _moods_cache, _triggers_cache, _cues
    _moods_cache = _triggers_cache = _cues = None
    return len(load_moods())


def detect_mood(text: str) -> str | None:
    """Pick a cue for a reply, or None to stay silent.

    Only the opening of the message is examined: that is where a reply states its verdict, and
    scanning the whole thing makes every long answer match something.
    """
    head = speakable(text)[:240]
    for mood, pattern in triggers():
        if pattern.search(head):
            return mood
    return None


def cue(mood: str) -> bool:
    """Play a pre-rendered Chatterbox cue. Returns False if that mood isn't in the bank.

    Cheap enough to use freely: it is a wav read, not a synthesis, so it costs nothing beyond
    the disk hit no matter how expressive it sounds.
    """
    if not core.VOICE_OUT or not core.VOICE_CUES:
        return False
    entry = cues().get(mood)
    if not entry:
        return False
    # Random variant: one fixed clip per mood becomes a soundboard within ten minutes, and a
    # jingle you have learned to ignore is worse than no cue at all.
    variants = entry.get("variants") or ([entry] if entry.get("file") else [])
    if not variants:
        return False
    if sp := _get_speaker():
        sp.interrupted.clear()
        sp._submit(("cue", phrasebank_dir() / random.choice(variants)["file"]))
        return True
    return False


def say(text: str, mood: str | None = None) -> None:
    """Speak a complete assistant message, optionally led by an emotional cue.

    Under Kokoro the two halves have wildly different costs, and that asymmetry is the whole
    reason cues exist: the cue carries the affect and is free (pre-rendered), while the body
    has to be synthesised live by a model with no emotion axis. Kokoro cannot sound urgent;
    the cue in front of it can. Under Chatterbox the body carries its own tone, so the cue is
    skipped - it would only repeat, a second late, what the reply already conveys.
    """
    if not core.VOICE_OUT:
        return
    # Detected unconditionally now, not only when cues are on: with an emotional engine the
    # mood also picks the tone the body is spoken in, which matters with cues off.
    if mood is None:
        mood = _pinned_mood or detect_mood(text)
    sp = _get_speaker()
    if not sp:
        return
    if mood and core.VOICE_CUES and not sp.engine.emotional:
        cue(mood)
    sp.say(text, mood)


def begin_stream() -> None:
    """Called from terminal_ui.stream_start so each streamed reply gets its own cue decision
    rather than inheriting the previous message's."""
    global _tagger
    _tagger = MoodTag(set(load_moods())) if core.VOICE_MOOD_TAGS else None
    if core.VOICE_OUT and _speaker is not None:
        _speaker.begin(_pinned_mood)


def feed_stream(chunk: str) -> None:
    if not core.VOICE_OUT:
        return
    sp = _get_speaker()
    if not sp:
        return
    if _tagger is not None:
        chunk = _tagger.feed(chunk)
        if _pinned_mood is None:   # an explicit /voice mood outranks the model's choice
            sp.set_mood(_tagger.mood)
        if not chunk:
            return                 # still deciding; nothing held back longer than a clause
    sp.feed(chunk)


def end_stream() -> None:
    if not core.VOICE_OUT:
        return
    if sp := _get_speaker():
        # A reply shorter than the tag window ends with text still held back.
        if _tagger is not None and (rest := _tagger.flush()):
            sp.feed(rest)
        sp.flush()


def interrupt() -> None:
    """Silence anything currently being spoken. Safe to call when voice is off."""
    if _speaker is not None:
        _speaker.stop()


# ---------------------------------------------------------------------------
# Input


def _get_stt() -> Any:
    global _stt_model
    with _lock:
        if _stt_model is None:
            from faster_whisper import WhisperModel

            _stt_model = WhisperModel(core.VOICE_STT_MODEL, device="cpu",
                                      compute_type="int8", cpu_threads=CPU_THREADS)
        return _stt_model


def _input_device() -> Any:
    dev = os.environ.get("VOICE_INPUT_DEVICE")
    return int(dev) if dev and dev.isdigit() else (dev or None)


def record_until_enter(prompt: str = "  [voice] recording - Enter to stop... ") -> Any:
    """Push-to-talk capture. Deliberately not VAD-gated: an endpointer that cuts you off
    mid-sentence is indistinguishable from a transcription bug, and getting that wrong is far
    more annoying than pressing a key."""
    import numpy as np
    import sounddevice as sd

    frames: queue.Queue[Any] = queue.Queue()
    stop = threading.Event()

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        device=_input_device(),
                        callback=lambda d, *a: frames.put(d.copy())):
        threading.Thread(target=lambda: (input(prompt), stop.set()), daemon=True).start()
        while not stop.is_set():
            time.sleep(0.05)

    chunks = []
    while not frames.empty():
        chunks.append(frames.get())
    return np.concatenate(chunks).flatten() if chunks else np.zeros(0, dtype="float32")


def transcribe(audio: Any) -> str:
    """Audio array -> text, or '' if there was nothing to hear."""
    import numpy as np

    if audio.size < RATE // 2:
        return ""
    peak = float(np.abs(audio).max())
    if peak < SILENCE_FLOOR:
        print(f"  [voice] silence (peak {peak:.4f}) - mic is not capturing. "
              f"Check levels: control mmsys.cpl,,1", file=sys.stderr)
        return ""
    segments, _ = _get_stt().transcribe(audio, beam_size=1, language="en", vad_filter=True)
    return " ".join(s.text for s in segments).strip()


def listen() -> str:
    """One spoken turn: record, transcribe, return the text ('' if nothing usable)."""
    ok, reason = available()
    if not ok:
        print(f"[voice] input disabled: {reason}", file=sys.stderr)
        core.VOICE_IN = False
        return ""
    interrupt()  # barge-in: talking over the assistant silences it
    t0 = time.perf_counter()
    audio = record_until_enter()
    text = transcribe(audio)
    if text:
        secs = audio.size / RATE
        print(f"  [voice] {secs:.1f}s heard in {time.perf_counter() - t0 - secs:.2f}s")
    return text


# ---------------------------------------------------------------------------
# /voice command


def _engine_line() -> str:
    """What is actually speaking, not what was asked for - the two differ whenever the
    Chatterbox server is down and the fallback quietly took over."""
    if _speaker is not None:
        eng = _speaker.engine
        return f"{eng.name} ({'live emotion' if eng.emotional else 'no emotion axis'})"
    return f"{core.VOICE_TTS_ENGINE} (loads on first word)"


def _cloned_voices() -> list[str] | None:
    """Voices the TTS server holds, or None if Chatterbox is not in play.

    Deliberately NOT read off the live speaker: the speaker is built lazily on the first
    spoken word, so before then every voice command would fall through to the Kokoro branch
    and silently set a Kokoro voice to a cloned name that Kokoro has never heard of.
    """
    if core.VOICE_TTS_ENGINE != "chatterbox":
        return None
    if _speaker is not None and _speaker.engine.name == "chatterbox":
        return _speaker.engine.available
    try:
        import requests

        r = requests.get(f"{core.VOICE_CBX_URL.rstrip('/')}/voices", timeout=4)
        r.raise_for_status()
        return r.json().get("voices") or ["default"]
    except Exception:
        return None


def _voice_line() -> str:
    if (avail := _cloned_voices()) is not None:
        return f"{core.VOICE_TTS_SPEAKER} (cloned, {len(avail)} available - /voice voices)"
    return f"{core.VOICE_TTS_VOICE} (kokoro)"


def status() -> str:
    ok, reason = available()
    lines = [
        f"voice input : {'on' if core.VOICE_IN else 'off'}",
        f"voice output: {'on' if core.VOICE_OUT else 'off'}",
        f"stt model   : {core.VOICE_STT_MODEL} (int8, {CPU_THREADS} threads, CPU)",
        f"tts engine  : {_engine_line()}",
        f"tts voice   : {_voice_line()}",
        f"mood        : {_pinned_mood or ('model-tagged' if core.VOICE_MOOD_TAGS else 'auto (trigger regexes)')}",
        f"cues        : {'on' if core.VOICE_CUES else 'off'}, {len(cues())} in bank",
        f"available   : {'yes' if ok else 'NO - ' + reason}",
    ]
    return "\n".join(lines)


def handle_command(arg: str) -> str:
    """Back /voice. Bare toggles both directions together."""
    # One declaration at the top rather than three at the branches that assign: Python
    # requires every `global` for a name to precede any USE of it in the function, and the
    # mood branch reads _speaker well before the voice branch would have declared it.
    global _speaker, _pinned_mood
    arg = arg.strip().lower()

    if arg in ("", "toggle"):
        turning_on = not (core.VOICE_IN or core.VOICE_OUT)
        if turning_on:
            ok, reason = available()
            if not ok:
                return f"voice unavailable: {reason}"
        core.VOICE_IN = core.VOICE_OUT = turning_on
        if not turning_on:
            interrupt()
        return f"voice {'on' if turning_on else 'off'} (input and output)"

    if arg == "status":
        return status()

    if arg in ("in", "out"):
        attr = "VOICE_IN" if arg == "in" else "VOICE_OUT"
        new = not getattr(core, attr)
        if new:
            ok, reason = available()
            if not ok:
                return f"voice unavailable: {reason}"
        setattr(core, attr, new)
        if arg == "out" and not new:
            interrupt()
        return f"voice {arg} {'on' if new else 'off'}"

    if arg == "stop":
        interrupt()
        return "stopped speaking"

    if arg.startswith("tts"):
        want = arg[3:].strip()
        if want not in ("kokoro", "chatterbox"):
            return (f"engine: {_engine_line()}\n"
                    "  /voice tts kokoro      no VRAM, no emotion at all (13Hz)\n"
                    "  /voice tts chatterbox  live emotion (102Hz), needs cbx_server on GPU")
        interrupt()
        core.VOICE_TTS_ENGINE = want
        with _lock:
            _speaker = None   # rebuilt lazily, so a bad switch costs nothing until spoken
        return f"tts engine -> {want} (loads on the next reply)"

    if arg.startswith("mood"):
        want = arg[4:].strip()
        if not want:
            return (f"pinned: {_pinned_mood or 'none - detected per reply'}\n"
                    "  /voice mood angry   hold that tone until cleared\n"
                    "  /voice mood off     back to per-reply detection")
        if want in ("off", "none", "clear"):
            _pinned_mood = None
            return "pinned mood cleared - back to detection"
        if want not in load_moods():
            return f"no mood {want!r} - /voice moods lists them"
        _pinned_mood = want
        exag, cfg = tone_for(want)
        flat = ""
        if _speaker is not None and not _speaker.engine.emotional:
            flat = "\n  kokoro cannot act on this - only the cue changes. /voice tts chatterbox"
        return f"pinned {want} (exaggeration {exag}, cfg {cfg}){flat}"

    if arg == "reload":
        n = reload_moods()
        return (f"reloaded {n} moods, {len(cues())} rendered, from {moods_file().name}\n"
                f"  triggers apply immediately; changed audio needs a re-render")

    if arg == "moods":
        bank = cues()
        if not bank:
            return (f"no phrasebank at {phrasebank_dir()}\n"
                    f"render one: .venv-cbx\\Scripts\\python.exe cbx_phrasebank.py")
        rows = []
        for m, e in bank.items():
            variants = e.get("variants") or ([e] if e.get("file") else [])
            trig = "auto" if e.get("trigger") else "manual"
            rows.append(f"  {m:<9} exag={e.get('exaggeration', '?'):<5} "
                        f"{len(variants)} variants  [{trig}]")
            rows += [f"             \"{v['text']}\"" for v in variants]
        rows.append(f"\n  tune: edit {moods_file()}, re-render, then /voice reload")
        return "\n".join(rows)

    if arg == "cues":
        core.VOICE_CUES = not core.VOICE_CUES
        return f"emotional cues {'on' if core.VOICE_CUES else 'off'}"

    if arg.startswith("cue "):
        mood = arg.removeprefix("cue ").strip()
        if not core.VOICE_OUT:
            return "voice output is off (/voice out)"
        return f"played {mood}" if cue(mood) else \
            f"no cue {mood!r} - see /voice moods"

    if arg == "voices":
        avail = _cloned_voices()
        if avail is None:
            if core.VOICE_TTS_ENGINE == "chatterbox":
                return f"tts server at {core.VOICE_CBX_URL} is not answering"
            return ("cloned voices are a chatterbox feature (/voice tts chatterbox)\n"
                    "  kokoro's own 54 voices are set with /voice voice NAME")
        return ("cloned voices: " + ", ".join(avail)
                + f"\n  current: {core.VOICE_TTS_SPEAKER}"
                + "\n  add more with voice-lab\\make_voices.py, then restart cbx_server")

    if arg.startswith("tags"):
        want = arg[4:].strip()
        if want in ("on", "off"):
            core.VOICE_MOOD_TAGS = want == "on"
        elif want:
            return "usage: /voice tags [on|off]"
        else:
            core.VOICE_MOOD_TAGS = not core.VOICE_MOOD_TAGS
        return (f"model-chosen moods {'on' if core.VOICE_MOOD_TAGS else 'off'}"
                + (" - takes effect on the next system prompt build"
                   if core.VOICE_MOOD_TAGS else
                   " - back to trigger regexes on the reply text"))

    if arg.startswith("voice "):
        name = arg.removeprefix("voice ").strip()
        # Chatterbox speakers are clones the server actually holds, so they can be validated;
        # Kokoro's 54 live in the model file and a wrong name fails loudly on its own.
        avail = _cloned_voices()
        if avail is not None:
            if name not in avail:
                return (f"no cloned voice {name!r}\n  have: " + ", ".join(avail))
            interrupt()
            core.VOICE_TTS_SPEAKER = name
            # The engine may not exist yet; when it does not, construction reads the core
            # value, so both orderings land on the same voice.
            if _speaker is not None and _speaker.engine.name == "chatterbox":
                _speaker.engine.voice = name
            return f"tts voice set to {name} (cloned)"
        core.VOICE_TTS_VOICE = name
        with _lock:
            if _speaker is not None:
                _speaker.voice = name
        return f"tts voice set to {name}"

    return ("usage: /voice [toggle] | status | in | out | stop\n"
            "       /voice moods | mood MOOD | cue MOOD | cues | reload | tags [on|off]\n"
            "       /voice tts kokoro|chatterbox | voices | voice NAME\n"
            "  bare /voice toggles input and output together")

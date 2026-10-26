#!/usr/bin/env python3
"""Low-latency Tunisian Derja ASR on top of faster-whisper (CTranslate2).

What this is (honestly): Whisper is an encoder-decoder model trained on <=30 s
windows; it is NOT a native streaming recognizer. This module gives a practical
"pseudo-streaming" experience:

  * Silero VAD (bundled with faster-whisper) finds speech onset and end of turn.
  * While a turn is in progress, the growing turn buffer is re-decoded every
    `step_s` seconds with greedy search -> PARTIAL hypotheses.
  * LocalAgreement-2: only the word prefix on which two consecutive partial
    decodes agree is marked "stable"; it never retracts. The rest is shown as
    "unstable" and may change.
  * When VAD sees `min_silence_ms` of silence, the turn is FINALIZED with a
    (configurable) beam search on the trimmed turn audio. Turns longer than
    `max_turn_s` are cut at the best VAD gap so the 30 s window is never exceeded.
  * Guards against silence hallucinations: no decode without VAD speech,
    minimum speech duration, faster-whisper no_speech/log-prob filters and a
    repeated-n-gram collapse.

Each partial re-decode costs one full encoder pass (Whisper pads to 30 s), so
the partial interval must be larger than the decode time on your hardware;
the engine adapts the interval automatically.

Usage (local machine with microphone):
    pip install faster-whisper "av<19" sounddevice soundfile numpy
    python live_asr.py --model ./whisper-tn-ct2                 # microphone
    python live_asr.py --model ./whisper-tn-ct2 --file talk.wav  # replay a file in real time
    python live_asr.py --model ./whisper-tn-ct2 --device cpu --compute-type int8
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

SR = 16000


@dataclass
class StreamConfig:
    language: str = "ar"
    step_s: float = 0.6            # minimum interval between partial decodes (audio seconds)
    min_silence_ms: int = 500      # silence that ends a turn (dominates end-of-turn latency)
    vad_threshold: float = 0.5
    preroll_s: float = 0.3         # audio kept before detected onset
    tail_pad_s: float = 0.2        # audio kept after last speech when finalizing
    max_turn_s: float = 25.0       # force a cut before Whisper's 30 s window
    min_speech_s: float = 0.3      # ignore blips shorter than this
    partial_beam: int = 1
    final_beam: int = 3
    emit_partials: bool = True
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    adaptive_step: bool = True     # widen step_s when decoding is slower than real time


def collapse_repeats(text: str, max_n: int = 4, max_rep: int = 2) -> str:
    """Collapse runs of the same 1..max_n-gram repeated more than max_rep times
    (typical Whisper looping hallucination)."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            out: List[str] = []
            while i < len(words):
                gram = words[i:i + n]
                reps = 1
                while words[i + reps * n:i + (reps + 1) * n] == gram and len(gram) == n:
                    reps += 1
                if reps > max_rep:
                    out.extend(gram)
                    i += reps * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return " ".join(words)


def common_prefix(a: List[str], b: List[str]) -> List[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def silero_vad_fn(threshold: float = 0.5) -> Callable[[np.ndarray], List[dict]]:
    """Returns f(audio)->[{'start','end'} in samples] using faster-whisper's Silero VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opts = VadOptions(threshold=threshold, min_speech_duration_ms=100,
                      min_silence_duration_ms=100, speech_pad_ms=0)

    def _vad(audio: np.ndarray) -> List[dict]:
        if len(audio) < SR // 10:
            return []
        return get_speech_timestamps(audio, vad_options=opts, sampling_rate=SR)
    return _vad


@dataclass
class Event:
    kind: str                  # "partial" | "final"
    text: str
    stable: str = ""
    unstable: str = ""
    audio_t: float = 0.0       # stream time (s) at which the event was produced
    turn_start: float = 0.0
    turn_end: float = 0.0
    compute_s: float = 0.0     # decode + VAD time spent producing this event


class TurnTranscriber:
    """Feed 16 kHz mono float32 chunks; receive partial/final events."""

    def __init__(self, model, cfg: Optional[StreamConfig] = None,
                 vad_fn: Optional[Callable[[np.ndarray], List[dict]]] = None):
        self.model = model
        self.cfg = cfg or StreamConfig()
        self.vad = vad_fn or silero_vad_fn(self.cfg.vad_threshold)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.t = 0.0                       # stream time of the end of the last fed sample
        self.ring = np.zeros(0, np.float32)
        self.turn: Optional[np.ndarray] = None
        self.turn_start = 0.0
        self.last_partial_t = 0.0
        self.prev_words: List[str] = []
        self.committed: List[str] = []
        self.step_s = self.cfg.step_s
        self.decode_times: List[float] = []

    # ---------------------------------------------------------------- decoding
    def _decode(self, audio: np.ndarray, beam: int, final: bool) -> str:
        kw = dict(language=self.cfg.language, task="transcribe", beam_size=beam,
                  condition_on_previous_text=False, vad_filter=False,
                  without_timestamps=True, no_speech_threshold=self.cfg.no_speech_threshold,
                  log_prob_threshold=self.cfg.log_prob_threshold)
        if not final:
            kw["temperature"] = 0.0          # no fallback loop for partials (latency)
        segments, _ = self.model.transcribe(audio, **kw)
        text = " ".join(s.text.strip() for s in segments).strip()
        return collapse_repeats(re.sub(r"\s+", " ", text))

    # ------------------------------------------------------------------- feed
    def feed(self, chunk: np.ndarray) -> List[Event]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.t += len(chunk) / SR
        events: List[Event] = []
        t0 = time.perf_counter()
        if self.turn is None:
            keep = int((self.cfg.preroll_s + 0.7) * SR)
            self.ring = np.concatenate([self.ring, chunk])[-keep:]
            tail = self.ring[-int(0.5 * SR):]
            if self.vad(tail):
                self.turn = self.ring[-int((self.cfg.preroll_s + 0.5) * SR):].copy()
                self.turn_start = self.t - len(self.turn) / SR
                self.last_partial_t = self.t
                self.prev_words, self.committed = [], []
                self.ring = np.zeros(0, np.float32)
            return events
        self.turn = np.concatenate([self.turn, chunk])
        turn_len = len(self.turn) / SR

        # end-of-turn check on the tail window only (cheap)
        win_s = self.cfg.min_silence_ms / 1000 + 0.6
        tail = self.turn[-int(win_s * SR):]
        spans = self.vad(tail)
        silence_s = (len(tail) - (spans[-1]["end"] if spans else 0)) / SR
        if silence_s >= self.cfg.min_silence_ms / 1000 and turn_len > win_s:
            events += self._finalize(t0)
            return events
        if turn_len >= self.cfg.max_turn_s:
            events += self._force_cut(t0)
            return events

        if self.cfg.emit_partials and self.t - self.last_partial_t >= self.step_s:
            d0 = time.perf_counter()
            words = self._decode(self.turn, self.cfg.partial_beam, final=False).split()
            dt = time.perf_counter() - d0
            self.decode_times.append(dt)
            if self.cfg.adaptive_step:   # keep partial decoding below real time
                self.step_s = max(self.cfg.step_s, 1.5 * dt)
            agreed = common_prefix(self.prev_words, words)
            if len(agreed) > len(self.committed) and agreed[:len(self.committed)] == self.committed:
                self.committed = agreed
            self.prev_words = words
            stable = " ".join(self.committed)
            unstable = " ".join(words[len(self.committed):]) if words[:len(self.committed)] == self.committed else " ".join(words)
            self.last_partial_t = self.t
            events.append(Event("partial", (stable + " " + unstable).strip(), stable, unstable,
                                audio_t=self.t, turn_start=self.turn_start,
                                compute_s=time.perf_counter() - t0))
        return events

    def _emit_final(self, audio: np.ndarray, start: float, end: float, t0: float) -> List[Event]:
        spans = self.vad(audio)
        speech = sum(s["end"] - s["start"] for s in spans) / SR
        text = ""
        if speech >= self.cfg.min_speech_s:
            text = self._decode(audio, self.cfg.final_beam, final=True)
        if not text:
            return []
        return [Event("final", text, text, "", audio_t=self.t, turn_start=start, turn_end=end,
                      compute_s=time.perf_counter() - t0)]

    def _finalize(self, t0: float) -> List[Event]:
        spans = self.vad(self.turn)
        last_end = spans[-1]["end"] if spans else len(self.turn)
        cut = min(len(self.turn), last_end + int(self.cfg.tail_pad_s * SR))
        audio, rest = self.turn[:cut], self.turn[cut:]
        start, end = self.turn_start, self.turn_start + last_end / SR
        self.turn = None
        self.ring = rest[-int((self.cfg.preroll_s + 0.7) * SR):]
        self.step_s = self.cfg.step_s
        return self._emit_final(audio, start, end, t0)

    def _force_cut(self, t0: float) -> List[Event]:
        look = int(4.0 * SR)
        tail = self.turn[-look:]
        spans = self.vad(tail)
        offset = len(self.turn) - len(tail)
        cut = len(self.turn)
        gaps = [(spans[i]["end"] + spans[i + 1]["start"]) // 2 for i in range(len(spans) - 1)]
        if gaps:
            cut = offset + gaps[-1]
        audio, rest = self.turn[:cut], self.turn[cut:]
        start = self.turn_start
        events = self._emit_final(audio, start, start + cut / SR, t0)
        # continue the same speech stream as a new turn
        self.turn = rest
        self.turn_start = start + cut / SR
        self.prev_words, self.committed = [], []
        self.last_partial_t = self.t
        return events

    def flush(self) -> List[Event]:
        if self.turn is None:
            return []
        return self._finalize(time.perf_counter())


def simulate_stream(model, audio: np.ndarray, cfg: Optional[StreamConfig] = None,
                    chunk_s: float = 0.1, vad_fn=None):
    """Replay `audio` as if it arrived in real time, using a virtual clock.

    Audio chunk k becomes available at wall time = its stream end time. The
    engine processes it as soon as it is available AND the previous call has
    finished, so slow decoding accumulates as queueing delay. Every event gets
    `emit_wall`, the virtual wall-clock time it would reach the user.
    Returns (events, total_compute_s)."""
    eng = TurnTranscriber(model, cfg, vad_fn)
    step = int(chunk_s * SR)
    wall = 0.0
    out = []
    total = 0.0
    for i in range(0, len(audio), step):
        chunk = audio[i:i + step]
        avail = (i + len(chunk)) / SR
        t0 = time.perf_counter()
        evs = eng.feed(chunk)
        dt = time.perf_counter() - t0
        total += dt
        wall = max(wall, avail) + dt
        for e in evs:
            out.append((wall, e))
    t0 = time.perf_counter()
    evs = eng.flush()
    dt = time.perf_counter() - t0
    total += dt
    wall = max(wall, len(audio) / SR) + dt
    out += [(wall, e) for e in evs]
    return out, total


# ---------------------------------------------------------------------- CLI
def _load_model(path: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(path, device=device, compute_type=compute_type)


def _print_event(e: Event):
    if e.kind == "partial":
        sys.stdout.write("\r\033[K… " + e.stable + (" \033[2m" + e.unstable + "\033[0m" if e.unstable else ""))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r\033[K[%.1f–%.1fs] %s\n" % (e.turn_start, e.turn_end, e.text))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="CTranslate2 model folder (faster-whisper format)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None, help="int8 (CPU) / int8_float16 (GPU) / float16 ...")
    ap.add_argument("--file", default=None, help="WAV/FLAC file to replay in real time instead of the mic")
    ap.add_argument("--step", type=float, default=0.6, help="partial decode interval (s)")
    ap.add_argument("--min-silence-ms", type=int, default=500)
    ap.add_argument("--final-beam", type=int, default=3)
    ap.add_argument("--no-partials", action="store_true")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    ctype = a.compute_type or ("int8_float16" if device == "cuda" else "int8")
    model = _load_model(a.model, device, ctype)
    cfg = StreamConfig(step_s=a.step, min_silence_ms=a.min_silence_ms,
                       final_beam=a.final_beam, emit_partials=not a.no_partials)
    eng = TurnTranscriber(model, cfg)
    print(f"Model loaded on {device} ({ctype}). Speak (Ctrl+C to stop)…", file=sys.stderr)

    if a.file:
        import soundfile as sf
        audio, sr = sf.read(a.file, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        step = int(0.1 * SR)
        t_start = time.perf_counter()
        for i in range(0, len(audio), step):
            target = t_start + (i + step) / SR
            time.sleep(max(0.0, target - time.perf_counter()))
            for e in eng.feed(audio[i:i + step]):
                _print_event(e)
        for e in eng.flush():
            _print_event(e)
        return

    import queue
    import sounddevice as sd
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(0.1 * SR), callback=cb):
        try:
            while True:
                chunk = q.get()
                while not q.empty() and len(chunk) < SR:   # catch up if decoding lagged
                    chunk = np.concatenate([chunk, q.get()])
                for e in eng.feed(chunk):
                    _print_event(e)
        except KeyboardInterrupt:
            for e in eng.flush():
                _print_event(e)


if __name__ == "__main__":
    main()

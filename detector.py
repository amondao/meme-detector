#!/usr/bin/env python3
"""
detector.py — ローカル版（効果音は自分にだけ聞こえる）
====================================================
PC のスピーカー出力（WASAPI ループバック）を録音し、faster-whisper で
リアルタイム文字起こし。登録キーワードを検出したら効果音をローカル再生する。

Discord 等の通話を「自分が聞いている音声そのまま」から拾うため、
E2EE(DAVE) やボットの制約を一切受けず、全員の声を 100% 取得できる。

構成:
    マイク/スピーカー → Engine(録音・無音区切り) → Transcriber(Whisper)
        → find_hit(キーワード検出) → SoundPlayer(効果音)

効果音を通話相手にも聞かせたい場合は meme_bot.py（bot再生版）を使う。
共有ロジックは core.py にある。
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from core import (
    APP_DIR, SOUNDS_DIR,
    WHISPER_SR, CHUNK_SEC, SILENCE_THRESH, SILENCE_SEC,
    MIN_UTTERANCE_SEC, MAX_BUF_SEC, MAX_PENDING_TASKS, NUM_WORKERS,
    Gate, add_ignore_phrases, default_output_name, find_hit,
    find_loopback_devices, has_real_speech, is_hallucination_text,
    load_config, normalize_mappings, rms, resample_to_16k, save_config,
    to_mono_f32,
)


# ================================================================
# デバイス選択（対話メニュー）
# ================================================================

def select_loopback_device(p: pyaudio.PyAudio, configured) -> dict | None:
    devices = find_loopback_devices(p)
    if not devices:
        print("エラー: WASAPI ループバックデバイスが見つかりません。")
        return None

    # 設定済みなら採用
    if configured is not None:
        dev = next((d for d in devices if d["index"] == configured), None)
        if dev:
            return dev
        print(f"[警告] 設定のデバイス {configured} が見つかりません。選択メニューを表示します。")

    # 自動推奨（既定の出力デバイスに一致するループバック）
    def_name = default_output_name(p)
    recommended = None
    print("\n[ループバックデバイス] あなたが聞いている音声（Discord 含む）")
    print("-" * 64)
    for d in devices:
        is_rec = bool(def_name and def_name[:15] in d["name"])
        mark = " ★ 推奨（既定の出力）" if is_rec else ""
        if is_rec and recommended is None:
            recommended = d["index"]
        print(f"  [{d['index']:>2}] {d['name']}  ({int(d['defaultSampleRate'])} Hz){mark}")
    print("-" * 64)

    prompt = "デバイス番号を入力"
    if recommended is not None:
        prompt += f"（推奨: {recommended}）"
    prompt += " [Enter=推奨]: "
    try:
        ans = input(prompt).strip()
    except EOFError:
        ans = ""
    if ans == "" and recommended is not None:
        chosen = recommended
    else:
        try:
            chosen = int(ans)
        except ValueError:
            print("無効な入力です。")
            return None
    return next((d for d in devices if d["index"] == chosen), None)


# ================================================================
# 効果音プレイヤー（ローカル再生）
# ================================================================

class SoundPlayer:
    """効果音をキューで順次再生する（事前にデコードしてキャッシュ）。"""

    def __init__(self, output_device, gate: "Gate | None" = None):
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._q: queue.Queue[tuple[str, float]] = queue.Queue()
        self._output_device = output_device
        self._gate = gate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _load(self, path: str) -> tuple[np.ndarray, int] | None:
        if path in self._cache:
            return self._cache[path]
        full = (APP_DIR / path) if not os.path.isabs(path) else Path(path)
        if not full.exists():
            print(f"[警告] 効果音が見つかりません: {full}", flush=True)
            return None
        try:
            data, sr = sf.read(str(full), dtype="float32", always_2d=False)
            self._cache[path] = (data, sr)
            return self._cache[path]
        except Exception as e:
            print(f"[警告] 効果音の読み込み失敗 ({path}): {e}", flush=True)
            return None

    def play(self, path: str, volume: float = 1.0):
        self._q.put((path, volume))

    def _run(self):
        while True:
            path, volume = self._q.get()
            loaded = self._load(path)
            if loaded is None:
                continue
            data, sr = loaded
            try:
                out = data if volume == 1.0 else (data * volume).astype(np.float32)
                if self._gate:
                    self._gate.begin()
                sd.play(out, samplerate=sr, device=self._output_device)
                sd.wait()
            except Exception as e:
                print(f"[警告] 効果音の再生失敗 ({path}): {e}", flush=True)
            finally:
                if self._gate:
                    self._gate.end()


# ================================================================
# 文字起こしワーカー
# ================================================================

class Transcriber:
    def __init__(self, model: WhisperModel, language: str,
                 mappings: list[dict], player: SoundPlayer,
                 cooldown_sec: float = 2.5, workers: int = NUM_WORKERS,
                 beam_size: int = 5):
        self.model     = model
        self.language  = language
        self.mappings  = mappings
        self.player    = player
        self.cooldown  = cooldown_sec
        self.beam_size = beam_size
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._last_fire: dict[str, float] = {}
        self._fire_lock = threading.Lock()  # 並列ワーカー間で cooldown を保護
        self._threads = [threading.Thread(target=self._run, daemon=True)
                         for _ in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def submit(self, audio: np.ndarray):
        # 詰まっている場合は古い音声を捨てて最新を優先
        if self._q.qsize() >= MAX_PENDING_TASKS:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put(audio)

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def _run(self):
        while True:
            audio = self._q.get()
            try:
                self._process(audio)
            except Exception as e:
                print(f"[エラー] 文字起こし失敗: {e}", file=sys.stderr, flush=True)

    def _process(self, audio: np.ndarray):
        # ① 音声エネルギーゲート：ほぼ無音なら文字起こしせず幻覚を防ぐ
        if not has_real_speech(audio):
            return

        segs, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segs).strip()
        if not text:
            return

        # ② 既知の幻覚定型句を除外
        if is_hallucination_text(text):
            return

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {text}", flush=True)

        # キーワード検出（クールダウン付き・並列ワーカー間で保護）
        with self._fire_lock:
            hit = find_hit(text, self.mappings, self._last_fire, self.cooldown,
                           time.monotonic())
        if hit:
            sound, vol = hit
            print(f"         🔊 {sound} (vol {vol})", flush=True)
            self.player.play(sound, vol)


# ================================================================
# 録音 → バッファ → 文字起こし
# ================================================================

class Engine:
    def __init__(self, transcriber: Transcriber, device: dict,
                 gate: "Gate | None" = None,
                 silence_sec: float = SILENCE_SEC, max_buf_sec: float = MAX_BUF_SEC):
        self.transcriber = transcriber
        self.gate = gate
        self.silence_sec = silence_sec
        self.max_buf_sec = max_buf_sec
        self.dev_index = device["index"]
        self.dev_sr    = int(device["defaultSampleRate"])
        self.dev_ch    = int(device["maxInputChannels"])
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()

    def _pa_callback(self, in_data, frame_count, time_info, status):
        mono = to_mono_f32(in_data, self.dev_ch)
        self._audio_q.put(resample_to_16k(mono, self.dev_sr))
        return (None, pyaudio.paContinue)

    def _buffer_loop(self):
        buffer = np.array([], dtype=np.float32)
        last_sound = time.monotonic()
        has_speech = False

        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.05)
            except queue.Empty:
                chunk = None

            # 効果音の再生中＋直後は録音を破棄（自分の効果音への反応ループ防止）
            if self.gate and self.gate.suppressed():
                buffer = np.array([], dtype=np.float32)
                has_speech = False
                last_sound = time.monotonic()
                continue

            if chunk is not None:
                buffer = np.concatenate([buffer, chunk])
                if rms(chunk) > SILENCE_THRESH:
                    last_sound = time.monotonic()
                    has_speech = True

            now = time.monotonic()
            buf_sec = len(buffer) / WHISPER_SR
            silence = now - last_sound

            if (has_speech and silence >= self.silence_sec) or buf_sec >= self.max_buf_sec:
                if buf_sec >= MIN_UTTERANCE_SEC and has_speech:
                    self.transcriber.submit(buffer.copy())
                buffer = np.array([], dtype=np.float32)
                has_speech = False
                last_sound = now

    def run(self):
        print("\n字幕＆ミーム検出を開始します。Ctrl+C で停止。\n", flush=True)
        worker = threading.Thread(target=self._buffer_loop, daemon=True)
        worker.start()

        chunk = int(self.dev_sr * CHUNK_SEC)
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.dev_ch,
                rate=self.dev_sr,
                input=True,
                input_device_index=self.dev_index,
                frames_per_buffer=chunk,
                stream_callback=self._pa_callback,
            )
            stream.start_stream()
            while stream.is_active() and not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n停止します。")
        finally:
            self._stop.set()
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()
            worker.join(timeout=3)


# ================================================================
# エントリポイント
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ミーム検出ツール（リアルタイム字幕＋効果音）")
    parser.add_argument("--model", default=None,
                        help="Whisper モデル (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", default=None, help="言語コード（既定 ja）")
    parser.add_argument("--device", type=int, default=None,
                        help="ループバックデバイスID（省略時は設定or選択）")
    args = parser.parse_args()

    cfg = load_config()
    model_size = args.model or cfg["model"]
    language   = args.language or cfg["language"]
    dev_cfg    = args.device if args.device is not None else cfg.get("loopback_device")

    SOUNDS_DIR.mkdir(exist_ok=True)

    # デバイス選択
    p = pyaudio.PyAudio()
    device = select_loopback_device(p, dev_cfg)
    p.terminate()
    if device is None:
        sys.exit(1)

    # 選択結果を設定に保存（次回から自動）
    if cfg.get("loopback_device") != device["index"]:
        cfg["loopback_device"] = device["index"]
        save_config(cfg)

    print(f"\nデバイス : {device['name']}")
    print(f"サンプルレート: {int(device['defaultSampleRate'])} Hz  チャンネル: {int(device['maxInputChannels'])}")

    # モデル読み込み
    workers = int(cfg.get("workers", NUM_WORKERS))
    print(f"Whisper モデル '{model_size}' を読み込み中...", flush=True)
    cpu_threads = cfg.get("cpu_threads") or min(8, os.cpu_count() or 4)
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=cpu_threads, num_workers=workers)
    print(f"モデル読み込み完了（CPUスレッド: {cpu_threads} / 並列: {workers}）。", flush=True)

    mappings = normalize_mappings(cfg)
    cooldown = cfg.get("cooldown_ms", 2500) / 1000.0
    add_ignore_phrases(cfg.get("ignore_phrases", []))
    all_kw = [kw for m in mappings for kw in m["keywords"]]
    print(f"登録効果音: {len(mappings)} 件 / キーワード {len(all_kw)} 個", flush=True)

    # 起動（gate で効果音の自己ループを防止）
    gate        = Gate()
    player      = SoundPlayer(cfg.get("output_device"), gate)
    transcriber = Transcriber(model, language, mappings, player, cooldown, workers,
                              beam_size=int(cfg.get("beam_size", 5)))
    engine      = Engine(transcriber, device, gate,
                         silence_sec=cfg.get("silence_sec", SILENCE_SEC),
                         max_buf_sec=cfg.get("max_buf_sec", MAX_BUF_SEC))
    engine.run()


if __name__ == "__main__":
    main()

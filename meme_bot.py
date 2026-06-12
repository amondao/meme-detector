#!/usr/bin/env python3
"""
ミーム検出ツール（Discord bot 再生版）
=====================================
1つのプログラムで以下を行うハイブリッド構成:

  ・聞く : ローカルのループバック録音（=相手の声）＋ マイク（=自分の声）
           → DAVE(E2EE) を完全回避して 100% 取得。話者を「自分/相手」で区別。
  ・流す : Discord bot のボイス接続経由で効果音を再生 → 通話相手全員に聞こえる。

bot は音声を「再生」するだけで「受信」はしないため、py-cord の DAVE 受信不具合
（約50%欠落）の影響を受けない。

コマンド:
  !join  : 自分が今いるボイスチャンネルに bot を呼び、録音＆検出を開始
  !leave : 退出・停止
  !status: 状態確認
  !who on|off : 個別の話者名表示 ⇔ 「自分/相手」表示の切替
  !kw list / !kw add <ファイル> <単語...> / !kw del <単語>

config.json に "token"（Bot Token）が必要。共有ロジックは core.py を参照。
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import discord
import discord.voice as dv
from discord.ext import commands
from faster_whisper import WhisperModel

# 共有ヘルパー・定数（core.py）
import core
from core import (
    APP_DIR, SOUNDS_DIR,
    WHISPER_SR, CHUNK_SEC, SILENCE_THRESH, SILENCE_SEC,
    MIN_UTTERANCE_SEC, MAX_BUF_SEC, MAX_PENDING_TASKS,
    rms, to_mono_f32, resample_to_16k, save_config,
)


# ================================================================
# 設定
# ================================================================

def load_config() -> dict:
    cfg = core.load_config()  # core の既定補完を流用
    # bot 用キーを補完
    cfg.setdefault("token", "YOUR_BOT_TOKEN_HERE")
    cfg.setdefault("mic_device", None)       # 自分の声（null=既定のマイク）
    cfg.setdefault("enable_mic", True)       # 自分の声を認識するか
    cfg.setdefault("enable_loopback", True)  # 相手の声を認識するか
    cfg.setdefault("identify_speakers", True)  # 個別の話者で字幕を出すか
    cfg.setdefault("anonymize_names", True)  # 実名でなく匿名通称（太郎/次郎…）で表示
    cfg.setdefault("self_user_id", None)     # 自分のDiscordユーザーID（相手の誤割当防止用・任意）
    return cfg


# ================================================================
# 文字起こしタスク（話者ラベル付き）
# ================================================================

class Task:
    __slots__ = ("label", "audio")

    def __init__(self, label: str, audio: np.ndarray):
        self.label = label
        self.audio = audio


# ================================================================
# 発話状態トラッカー（誰がいつ喋っていたかを記録）
# ================================================================

# 話者の匿名通称（実名を出さないための置き換え。登場順に割り当てる）
SPEAKER_ALIASES = ["太郎", "次郎", "三郎", "四郎", "五郎",
                   "六郎", "七郎", "八郎", "九郎", "十郎"]


class SpeakingTracker:
    """
    SpeakingSink（パケット到着ベースの発話検出）から
    「どのユーザーがいつ喋っていたか」を区間として記録する。

    bot は音声を復号できない(DAVE)が、発話タイミングは取得できるため、
    ループバックで拾った実音声の時間窓に重なる話者を特定できる。

    lead     : パケット到着がループバック再生より先行する分の時間補正
    margin   : タイミングの揺らぎを吸収する余白
    exclude  : 割当から除外する uid（本人・bot。ループバックに本人の声は
               入らないため、本人の相槌が他人の発話を奪うのを防ぐ）
    anonymize: True なら実名でなく匿名通称（太郎/次郎…）で表示する
    """
    def __init__(self, margin: float = 0.4, lead: float = 0.15,
                 exclude: set[int] | None = None, anonymize: bool = True):
        self._lock = threading.Lock()
        self._active: dict[int, float] = {}          # uid -> 発話開始(monotonic)
        self._intervals: deque = deque(maxlen=400)   # (uid, start, end)
        self._names: dict[int, str] = {}
        self._aliases: dict[int, str] = {}           # uid -> 匿名通称
        self.margin = margin
        self.lead = lead
        self.exclude = exclude or set()
        self.anonymize = anonymize

    def _label_for(self, uid: int) -> str:
        """uid を表示用ラベルに変換（匿名時は登場順の通称を割り当て）。要ロック保持。"""
        if not self.anonymize:
            return self._names.get(uid, f"User{uid}")
        alias = self._aliases.get(uid)
        if alias is None:
            n = len(self._aliases)
            alias = SPEAKER_ALIASES[n] if n < len(SPEAKER_ALIASES) else f"話者{n + 1}"
            self._aliases[uid] = alias
        return alias

    def update(self, uid: int, name: str | None, speaking: bool):
        now = time.monotonic()
        with self._lock:
            if name:
                self._names[uid] = name
            if speaking:
                self._active.setdefault(uid, now)
            else:
                start = self._active.pop(uid, None)
                if start is not None:
                    self._intervals.append((uid, start, now))

    def dominant_active(self, min_active: float = 0.15) -> int | None:
        """現在発話中の主話者（最も長く話し続けている人）の uid。除外IDは無視。"""
        now = time.monotonic()
        with self._lock:
            cands = [(s, uid) for uid, s in self._active.items()
                     if uid not in self.exclude and now - s >= min_active]
        return min(cands)[1] if cands else None

    def who_spoke(self, t_start: float, t_end: float) -> str | None:
        """
        発話窓に最も重なった話者名を返す。
        重なりが小さい・複数人が拮抗して曖昧な場合は None（→「相手」表示）。
        誤った名前を出すより安全側に倒す。
        """
        a = t_start - self.lead - self.margin
        b = t_end - self.lead + self.margin
        now = time.monotonic()
        with self._lock:
            ivs = list(self._intervals)
            ivs += [(uid, s, now) for uid, s in self._active.items()]
            overlap: dict[int, float] = {}
            for uid, s, e in ivs:
                if uid in self.exclude:
                    continue
                ov = min(b, e) - max(a, s)
                if ov > 0:
                    overlap[uid] = overlap.get(uid, 0.0) + ov
            if not overlap:
                return None
            uid, best = max(overlap.items(), key=lambda kv: kv[1])
            total = sum(overlap.values())
            if best < 0.1 or best / total < 0.55:
                return None
            return self._label_for(uid)


# ================================================================
# 発話検出専用 Sink（音声は使わず、発話状態だけを取得）
# ================================================================

class SpeakingSink(discord.sinks.Sink):
    """
    音声そのものは使わず（実音声はループバックで取得）、py-cord の SpeakingTimer が
    RTP パケット到着から判定する「発話開始/終了」イベントだけを SpeakingTracker に流す。

    SpeakingTimer は 0.2 秒パケットが来なければ自動で発話終了を出すため、
    Discord の op5 発話イベント（終了が信頼できない）より遥かに正確。
    is_opus()=True でデコードを完全にスキップし、DAVE 復号失敗も CPU 浪費も回避する。
    """
    # SinkEventRouter が参照: (イベント名, メソッド名)
    __sink_listeners__ = [
        ("on_member_speaking_start", "_on_speaking_start"),
        ("on_member_speaking_stop", "_on_speaking_stop"),
    ]

    def __init__(self, tracker: SpeakingTracker):
        super().__init__()
        self.tracker = tracker

    def walk_children(self, *args, **kwargs):
        return []

    def is_opus(self) -> bool:
        return True   # デコードしない（発話検出が目的）

    def write(self, data, user):
        pass          # 音声は破棄（ループバックで取得済み）

    def cleanup(self):
        pass

    def _on_speaking_start(self, member):
        if member is not None:
            self.tracker.update(member.id, getattr(member, "display_name", None), True)

    def _on_speaking_stop(self, member):
        if member is not None:
            self.tracker.update(member.id, getattr(member, "display_name", None), False)


# ================================================================
# 文字起こしワーカー（キーワード検出 → bot 再生をスケジュール）
# ================================================================

class Transcriber:
    def __init__(self, model: WhisperModel, language: str, mappings: list[dict],
                 play_q: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                 cooldown_sec: float = 2.5, workers: int = core.NUM_WORKERS):
        self.model    = model
        self.language = language
        self.mappings = mappings
        self.cooldown = cooldown_sec
        self.play_q   = play_q
        self.loop     = loop
        self._q: queue.Queue[Task] = queue.Queue()
        self._last_fire: dict[str, float] = {}
        self._fire_lock = threading.Lock()  # 並列ワーカー間で cooldown を保護
        self._threads = [threading.Thread(target=self._run, daemon=True)
                         for _ in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def submit(self, task: Task):
        if self._q.qsize() >= MAX_PENDING_TASKS:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put(task)

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def _run(self):
        while True:
            task = self._q.get()
            try:
                self._process(task)
            except Exception as e:
                print(f"[エラー] 文字起こし失敗: {e}", file=sys.stderr, flush=True)

    def _process(self, task: Task):
        # ① 音声エネルギーゲート：ほぼ無音なら文字起こしせず幻覚を防ぐ
        if not core.has_real_speech(task.audio):
            return

        segs, _ = self.model.transcribe(
            task.audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        text = "".join(s.text for s in segs).strip()
        if not text:
            return

        # ② 既知の幻覚定型句を除外
        if core.is_hallucination_text(text):
            return

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {task.label}: {text}", flush=True)

        with self._fire_lock:
            hit = core.find_hit(text, self.mappings, self._last_fire, self.cooldown,
                                time.monotonic())
        if hit:
            sound, vol = hit
            print(f"         🔊 {sound} (vol {vol})（全員に再生）", flush=True)
            # bot のイベントループに再生を依頼
            asyncio.run_coroutine_threadsafe(self.play_q.put((sound, vol)), self.loop)


# ================================================================
# 音声キャプチャ（ループバック＝相手 / マイク＝自分 を別バッファで処理）
# ================================================================

class LabeledCapture:
    """1つの入力デバイスを録音し、無音区切りで Transcriber にラベル付き投入。"""

    def __init__(self, transcriber: Transcriber, label: str,
                 dev_index: int, dev_sr: int, dev_ch: int,
                 gate: "core.Gate | None" = None,
                 silence_sec: float = SILENCE_SEC, max_buf_sec: float = MAX_BUF_SEC,
                 tracker: "SpeakingTracker | None" = None):
        self.transcriber = transcriber
        self.label = label
        self.dev_index = dev_index
        self.dev_sr = dev_sr
        self.dev_ch = dev_ch
        self.gate = gate
        self.silence_sec = silence_sec
        self.max_buf_sec = max_buf_sec
        self.tracker = tracker   # セットされていれば発話区間から話者名を解決
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._buf_thread = threading.Thread(target=self._buffer_loop, daemon=True)

    def _callback(self, in_data, frame_count, time_info, status):
        mono = to_mono_f32(in_data, self.dev_ch)
        self._q.put(resample_to_16k(mono, self.dev_sr))
        return (None, pyaudio.paContinue)

    def _buffer_loop(self):
        buffer = np.array([], dtype=np.float32)
        last_sound = time.monotonic()
        has_speech = False
        utt_start = None  # 発話開始時刻（話者解決の時間窓に使用）
        turn_uid = None   # この発話の主話者 uid（交代を検知したら早期区切り）

        def flush(end_t: float):
            nonlocal buffer, has_speech, utt_start, turn_uid, last_sound
            buf_sec = len(buffer) / WHISPER_SR
            if buf_sec >= MIN_UTTERANCE_SEC and has_speech:
                label = self.label
                # 発話区間（無音待ちを含まない実発話の窓）で話者名を解決。
                # 曖昧なら既定ラベル（相手）のまま
                if self.tracker is not None and utt_start is not None:
                    who = self.tracker.who_spoke(utt_start, end_t)
                    if who:
                        label = who
                self.transcriber.submit(Task(label, buffer.copy()))
            buffer = np.array([], dtype=np.float32)
            has_speech = False
            utt_start = None
            turn_uid = None
            last_sound = time.monotonic()

        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.05)
            except queue.Empty:
                chunk = None
            # 効果音の再生中＋直後は録音を破棄（自分の効果音への反応ループ防止）
            if self.gate and self.gate.suppressed():
                buffer = np.array([], dtype=np.float32)
                has_speech = False
                utt_start = None
                turn_uid = None
                last_sound = time.monotonic()
                continue
            if chunk is not None:
                buffer = np.concatenate([buffer, chunk])
                if rms(chunk) > SILENCE_THRESH:
                    last_sound = time.monotonic()
                    if not has_speech:
                        utt_start = last_sound   # 発話の立ち上がり
                    has_speech = True

            now = time.monotonic()
            buf_sec = len(buffer) / WHISPER_SR

            # 話者交代を検知したら早期に区切る（A→B の連続発話が混ざるのを防ぐ）
            if (self.tracker is not None and has_speech
                    and buf_sec >= MIN_UTTERANCE_SEC):
                dom = self.tracker.dominant_active()
                if dom is not None:
                    if turn_uid is None:
                        turn_uid = dom
                    elif dom != turn_uid:
                        flush(last_sound)
                        continue

            if (has_speech and now - last_sound >= self.silence_sec) or buf_sec >= self.max_buf_sec:
                flush(last_sound)

    def start(self, p: pyaudio.PyAudio):
        chunk = int(self.dev_sr * CHUNK_SEC)
        self._stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.dev_ch,
            rate=self.dev_sr,
            input=True,
            input_device_index=self.dev_index,
            frames_per_buffer=chunk,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self._buf_thread.start()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass


class CaptureManager:
    """ループバック＋マイクの複数キャプチャをまとめて管理。"""

    def __init__(self):
        self._p: pyaudio.PyAudio | None = None
        self._caps: list[LabeledCapture] = []

    def start(self, transcriber: Transcriber, cfg: dict,
              gate: "core.Gate | None" = None,
              tracker: "SpeakingTracker | None" = None):
        self._p = pyaudio.PyAudio()
        started = []
        silence_sec = cfg.get("silence_sec", SILENCE_SEC)
        max_buf_sec = cfg.get("max_buf_sec", MAX_BUF_SEC)

        # 相手の声（ループバック）— 効果音の自己ループ防止のため必ず gate を適用
        # tracker があれば発話状態から個別の話者名を解決する
        if cfg.get("enable_loopback", True):
            dev = _resolve_loopback(self._p, cfg.get("loopback_device"))
            if dev:
                cap = LabeledCapture(transcriber, "相手", dev["index"],
                                     int(dev["defaultSampleRate"]), int(dev["maxInputChannels"]),
                                     gate=gate, silence_sec=silence_sec, max_buf_sec=max_buf_sec,
                                     tracker=tracker)
                cap.start(self._p)
                self._caps.append(cap)
                started.append(f"相手←{dev['name']}")

        # 自分の声（マイク）— スピーカー利用時の回り込みに備え gate を適用
        if cfg.get("enable_mic", True):
            dev = _resolve_mic(self._p, cfg.get("mic_device"))
            if dev:
                ch = min(int(dev["maxInputChannels"]), 2)
                cap = LabeledCapture(transcriber, "自分", dev["index"],
                                     int(dev["defaultSampleRate"]), ch,
                                     gate=gate, silence_sec=silence_sec, max_buf_sec=max_buf_sec)
                cap.start(self._p)
                self._caps.append(cap)
                started.append(f"自分←{dev['name']}")

        return started

    def stop(self):
        for c in self._caps:
            c.stop()
        self._caps.clear()
        if self._p is not None:
            self._p.terminate()
            self._p = None


def _resolve_loopback(p: pyaudio.PyAudio, configured) -> dict | None:
    devices = list(p.get_loopback_device_info_generator())
    if not devices:
        print("[警告] ループバックデバイスが見つかりません（相手の声は取得できません）。", flush=True)
        return None
    if configured is not None:
        d = next((d for d in devices if d["index"] == configured), None)
        if d:
            return d
    # 既定の出力に一致するものを自動選択
    def_name = core.default_output_name(p)
    d = next((d for d in devices if def_name and def_name[:15] in d["name"]), None)
    return d or devices[0]


def _resolve_mic(p: pyaudio.PyAudio, configured) -> dict | None:
    if configured is not None:
        try:
            return p.get_device_info_by_index(configured)
        except Exception:
            pass
    # WASAPI 既定の入力
    try:
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if info["type"] == pyaudio.paWASAPI:
                return p.get_device_info_by_index(info["defaultInputDevice"])
    except Exception:
        pass
    print("[警告] マイクが見つかりません（自分の声は取得できません）。", flush=True)
    return None


# ================================================================
# 効果音プレイヤー（bot のボイス接続で再生）
# ================================================================

async def sound_player(vc: dv.VoiceClient, play_q: asyncio.Queue,
                       gate: "core.Gate | None" = None):
    while True:
        sound, volume = await play_q.get()
        full = (APP_DIR / sound) if not os.path.isabs(sound) else Path(sound)
        if not full.exists():
            print(f"[警告] 効果音が見つかりません: {full}", flush=True)
            continue
        # 直前の再生が終わるまで待つ
        while vc.is_playing():
            await asyncio.sleep(0.05)
        try:
            source = discord.PCMVolumeTransformer(
                discord.FFmpegPCMAudio(str(full)), volume=volume
            )
            # 再生中は検出を止める（再生終了時に after で tail 付きで解除）
            if gate:
                gate.begin()

            def _after(err, g=gate):
                if g:
                    g.end()

            vc.play(source, after=_after)
        except Exception as e:
            print(f"[警告] 効果音の再生失敗: {e}", flush=True)
            if gate:
                gate.end()


# ================================================================
# Bot
# ================================================================

cfg = load_config()

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states    = True
intents.members         = True

bot = commands.Bot(command_prefix="!", intents=intents)

_model:       WhisperModel | None  = None
_transcriber: Transcriber | None   = None
_capture:     CaptureManager | None = None
_play_q:      asyncio.Queue | None  = None
_player_task: asyncio.Task | None   = None
_gate:        core.Gate | None      = None
_speaking:    SpeakingTracker | None = None
_speaking_sink = None


@bot.event
async def on_ready():
    global _model, _play_q, _gate, _speaking
    print(f"\nBot ログイン: {bot.user}", flush=True)
    print(f"Whisper モデル '{cfg['model']}' を読み込み中...", flush=True)
    cpu_threads = cfg.get("cpu_threads") or min(8, os.cpu_count() or 4)
    workers = int(cfg.get("workers", core.NUM_WORKERS))
    _model = WhisperModel(cfg["model"], device="cpu", compute_type="int8",
                          cpu_threads=cpu_threads, num_workers=workers)
    print(f"モデル読み込み完了（CPUスレッド: {cpu_threads} / 並列: {workers}）。", flush=True)
    _play_q = asyncio.Queue()
    _gate = core.Gate()
    # 個別話者の特定用トラッカー（自分のIDは相手音声への誤割当を避けるため除外）
    exclude = set()
    if cfg.get("self_user_id"):
        try:
            exclude.add(int(cfg["self_user_id"]))
        except (TypeError, ValueError):
            pass
    _speaking = SpeakingTracker(exclude=exclude,
                                anonymize=cfg.get("anonymize_names", True))
    core.add_ignore_phrases(cfg.get("ignore_phrases", []))
    SOUNDS_DIR.mkdir(exist_ok=True)
    n_map = len(core.normalize_mappings(cfg))
    print(f"登録効果音: {n_map} 件", flush=True)
    mode = "個別話者" if cfg.get("identify_speakers", True) else "相手まとめて"
    print(f"字幕の話者表示: {mode}", flush=True)
    print("準備完了。ボイスチャンネルに入って !join。\n", flush=True)


@bot.command(name="join", aliases=["j"])
async def cmd_join(ctx: commands.Context):
    global _transcriber, _capture, _player_task, _speaking_sink

    if ctx.author.voice is None:
        await ctx.send("先にボイスチャンネルに入ってください。")
        return
    if _model is None:
        await ctx.send("モデル読み込み中です。少し待ってください。")
        return

    channel = ctx.author.voice.channel
    if ctx.voice_client is not None:
        if ctx.voice_client.is_listening() or ctx.voice_client.is_recording():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ctx.voice_client.stop_listening()
        await ctx.voice_client.move_to(channel)
        vc = ctx.voice_client
    else:
        vc = await channel.connect(cls=dv.VoiceClient)

    # 既存の検出を止めてから再構築
    if _capture is not None:
        _capture.stop()
    loop = asyncio.get_event_loop()
    mappings = core.normalize_mappings(cfg)
    cooldown = cfg.get("cooldown_ms", 2500) / 1000.0
    workers = int(cfg.get("workers", core.NUM_WORKERS))
    _transcriber = Transcriber(_model, cfg["language"], mappings, _play_q, loop, cooldown, workers)
    _capture = CaptureManager()
    tracker = _speaking if cfg.get("identify_speakers", True) else None
    if tracker is not None:
        # !join した本人＝このツールの利用者。ループバックに本人の声は入らないので、
        # 本人の発話（相槌等）が他人の発話に誤割当されるのを防ぐため除外する
        tracker.exclude.add(ctx.author.id)
        if bot.user:
            tracker.exclude.add(bot.user.id)
    started = _capture.start(_transcriber, cfg, _gate, tracker)

    # 個別話者の特定: 発話状態を取るため軽量 Sink で listen（音声は使わない）
    if tracker is not None:
        try:
            _speaking_sink = SpeakingSink(_speaking)
            _speaking_sink.vc = vc   # reader は init(vc) を呼ばないので手動セット
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vc.start_listening(_speaking_sink)
        except Exception as e:
            _speaking_sink = None
            print(f"[警告] 発話状態の取得を開始できませんでした（話者は『相手』表示）: {e}",
                  flush=True)

    # 再生タスクは現在の vc / gate で作り直す（再 join 時に古い vc を参照しないように）
    if _player_task is not None and not _player_task.done():
        _player_task.cancel()
    _player_task = asyncio.create_task(sound_player(vc, _play_q, _gate))

    src = "\n".join(f"・{s}" for s in started) if started else "（音声ソースなし）"
    await ctx.send(
        f"**{channel.name}** で検出開始。\n{src}\n"
        f"モデル: `{cfg['model']}` / 言語: `{cfg['language']}`\n"
        f"効果音: {len(mappings)} 件登録"
    )
    print(f"=== 検出開始: {channel.name} ===", flush=True)
    for s in started:
        print(f"  {s}", flush=True)
    print(flush=True)


@bot.command(name="leave", aliases=["l"])
async def cmd_leave(ctx: commands.Context):
    global _capture, _speaking_sink
    if _capture is not None:
        _capture.stop()
        _capture = None
    if ctx.voice_client is not None:
        if ctx.voice_client.is_listening() or ctx.voice_client.is_recording():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ctx.voice_client.stop_listening()
        await ctx.voice_client.disconnect()
    _speaking_sink = None
    await ctx.send("退出しました。")
    print("=== 検出停止 ===", flush=True)


@bot.command(name="status")
async def cmd_status(ctx: commands.Context):
    vc = ctx.voice_client
    ch = vc.channel.name if vc and vc.is_connected() else "未接続"
    active = _capture is not None and len(_capture._caps) > 0
    pending = _transcriber.pending if _transcriber else 0
    n_map = len(core.normalize_mappings(cfg))
    await ctx.send(
        f"**状態**\nチャンネル: {ch}\n"
        f"検出: {'🔴 ON' if active else '⚫ OFF'}\n"
        f"文字起こし待ち: {pending} 件\n"
        f"モデル: `{cfg['model']}` / 言語: `{cfg['language']}`\n"
        f"効果音: {n_map} 件登録"
    )


# ---- !who ----

@bot.command(name="who")
async def cmd_who(ctx: commands.Context, mode: str = ""):
    """個別話者表示の ON/OFF を即時切り替える。"""
    mode = mode.lower().strip()
    if mode not in ("on", "off"):
        await ctx.send("`!who on`（個別の名前で表示）/ `!who off`（相手とまとめて表示）")
        return
    on = mode == "on"
    cfg["identify_speakers"] = on
    save_config(cfg)
    if _capture is not None:
        for cap in _capture._caps:
            if cap.label == "相手":
                cap.tracker = _speaking if on else None
    await ctx.send(f"個別話者表示: {'ON' if on else 'OFF（相手とまとめて表示）'}")


def _ensure_mappings() -> list:
    """cfg を mappings 形式に統一して返す（旧 keywords は変換して取り込む）。"""
    if "mappings" not in cfg:
        cfg["mappings"] = core.normalize_mappings(cfg)
        cfg.pop("keywords", None)
    return cfg["mappings"]


def _apply_mappings():
    """編集後のマッピングを保存し、稼働中の Transcriber に反映。"""
    save_config(cfg)
    if _transcriber:
        _transcriber.mappings = core.normalize_mappings(cfg)


@bot.group(name="kw")
async def cmd_kw(ctx: commands.Context):
    if ctx.invoked_subcommand is None:
        await ctx.send("`!kw list` / `!kw add <ファイル> <単語...>` / `!kw del <単語>`")


@cmd_kw.command(name="list")
async def kw_list(ctx: commands.Context):
    maps = _ensure_mappings()
    if not maps:
        await ctx.send("効果音マッピング未設定。")
        return
    lines = [f"・`{m['file']}` (vol {m.get('volume',1.0)}): {', '.join(m['keywords'])}"
             for m in maps]
    # Discord の 2000 文字制限対策で分割
    msg = "**効果音マッピング**\n"
    for ln in lines:
        if len(msg) + len(ln) > 1900:
            await ctx.send(msg); msg = ""
        msg += ln + "\n"
    if msg:
        await ctx.send(msg)


@cmd_kw.command(name="add")
async def kw_add(ctx: commands.Context, sound_file: str, *words):
    """既存ファイルのマッピングに単語を追加。なければ新規作成。"""
    if not words:
        await ctx.send("単語を1つ以上指定してください。例: `!kw add sounds/laugh.mp3 笑 わら`")
        return
    maps = _ensure_mappings()
    target = next((m for m in maps if m["file"] == sound_file), None)
    if target is None:
        target = {"keywords": [], "file": sound_file, "volume": 1.0}
        maps.append(target)
    for w in words:
        if w not in target["keywords"]:
            target["keywords"].append(w)
    _apply_mappings()
    await ctx.send(f"追加: `{sound_file}` ← {', '.join(words)}")


@cmd_kw.command(name="del")
async def kw_del(ctx: commands.Context, word: str):
    """指定単語を全マッピングから削除（空になったマッピングは除去）。"""
    maps = _ensure_mappings()
    removed = False
    for m in list(maps):
        if word in m["keywords"]:
            m["keywords"].remove(word)
            removed = True
            if not m["keywords"]:
                maps.remove(m)
    if not removed:
        await ctx.send(f"`{word}` は未登録。")
        return
    _apply_mappings()
    await ctx.send(f"削除: `{word}`")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR,
                        format="%(levelname)s:%(name)s: %(message)s")
    # 音声受信のデコード/復号エラーは無害（復号音声は使わずループバックで取得し、
    # 発話タイミングだけ利用）。大量のトレースバックを抑制する。
    for noisy in ("discord.voice.receive.reader",
                  "discord.voice.receive.router",
                  "discord.voice.gateway"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    if cfg["token"] in ("", "YOUR_BOT_TOKEN_HERE"):
        print("config.json の token を設定してください。")
        sys.exit(1)
    bot.run(cfg["token"])

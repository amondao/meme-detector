#!/usr/bin/env python3
"""
core.py — 共有ロジック
======================
detector.py（ローカル版）と meme_bot.py（bot再生版）の両方から使われる
共通部分。アプリ本体のロジックはここには置かない。

含まれるもの:
  - 定数（音声処理・検出のチューニング値）
  - 設定ファイル (config.json) の読み書きと効果音マッピングの正規化
  - キーワード検出 (find_hit) とクールダウン
  - 音声ユーティリティ（RMS / モノラル化 / 16kHz リサンプリング）
  - Whisper の幻覚（ハルシネーション）対策フィルタ
  - Gate: 効果音再生中に検出を止める自己ループ防止機構
  - WASAPI ループバックデバイスの列挙ヘルパー
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

# ================================================================
# 定数（チューニング値）
# ================================================================
WHISPER_SR        = 16000   # Whisper 入力サンプルレート
CHUNK_SEC         = 0.1     # 録音の取得ブロック単位（秒）
SILENCE_THRESH    = 0.006   # 無音判定の RMS しきい値
SILENCE_SEC       = 0.5     # 無音がこの秒数続いたら文字起こし（小さいほど低遅延）
MIN_UTTERANCE_SEC = 0.3     # これより短い音声は無視
MAX_BUF_SEC       = 5.0     # バッファ上限（秒）。長い発話もこの間隔で区切る
MAX_PENDING_TASKS = 2       # 文字起こし待ち上限（超えたら古い音声を破棄）
NUM_WORKERS       = 2       # 文字起こしの並列ワーカー数（複数話者の同時発話を捌く）
PLAYBACK_TAIL_SEC = 1.0     # 効果音の再生終了後も検出を止め続ける秒数（残響対策）

# --- 幻覚（ハルシネーション）対策 ---
MIN_SPEECH_RATIO  = 0.12    # バッファ中この割合以上が有音でなければ文字起こししない
SPEECH_PEAK_MIN   = 0.015   # バッファ中の最大音量がこれ未満なら無音とみなす

# パス（このファイルと同じディレクトリを基準にする）
APP_DIR     = Path(__file__).parent
CONFIG_PATH = APP_DIR / "config.json"
SOUNDS_DIR  = APP_DIR / "sounds"

# Whisper が無音・ノイズに対して頻出する定型幻覚（記号除去後の完全一致で除外）
# ※「お疲れ様」等はキーワード登録されうるので含めない（エネルギーゲートで対処）
HALLUCINATION_PHRASES = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございました",
    "ご覧いただきありがとうございました",
    "ご清聴ありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "高評価とチャンネル登録をよろしくお願いします",
    "次の動画でお会いしましょう",
    "またね",
    "では次の動画でお会いしましょう",
}

DEFAULT_CONFIG = {
    "model": "small",
    "language": "ja",
    "device": "auto",          # auto/cpu/cuda（GPUがあれば自動でcuda）
    "compute_type": "auto",    # auto/int8/float16 等
    "loopback_device": None,   # null=自動選択（既定の再生デバイス）
    "output_device": None,     # 効果音の再生先（null=既定の出力）
    "cooldown_ms": 2500,       # 同じ効果音が連続で鳴るのを抑制するミリ秒
    "beam_size": 5,            # Whisper のビーム幅。大きいほど高精度・低速（1で最速）
    "mappings": [
        {"keywords": ["草", "くさ"], "file": "sounds/kusa.wav", "volume": 1.0},
        {"keywords": ["おめでとう"], "file": "sounds/fanfare.wav", "volume": 1.0},
        {"keywords": ["爆発"], "file": "sounds/explosion.wav", "volume": 1.0}
    ]
}


# ================================================================
# 設定ファイル
# ================================================================

def load_config() -> dict:
    """config.json を読み込む（無ければ既定値で作成）。不足キーは既定で補完。"""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"config.json を作成しました: {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def pick_device(cfg: dict) -> tuple[str, str]:
    """
    Whisper を動かすデバイスと計算精度を決める。
    config の "device" / "compute_type" が "auto"（既定）なら、
    GPU(CUDA) が使えれば cuda/float16、無ければ cpu/int8 を自動選択する。

    GPU が使えると medium / large-v3 でも高速（CPU の数倍速）なので、
    精度と速度を両立できる。
    """
    device = cfg.get("device", "auto")
    compute = cfg.get("compute_type", "auto")
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def normalize_mappings(cfg: dict) -> list[dict]:
    """
    設定の効果音マッピングを統一形式に正規化する。
      新形式: "mappings": [{"keywords": [...], "file": "...", "volume": 1.0}, ...]
      旧形式: "keywords": {"単語": "ファイル"}  → 単語1個・音量1.0 として変換
    """
    out: list[dict] = []
    if cfg.get("mappings"):
        for m in cfg["mappings"]:
            kws = [k for k in m.get("keywords", []) if k]
            if not kws or not m.get("file"):
                continue
            out.append({
                "keywords": kws,
                "file": m["file"],
                "volume": float(m.get("volume", 1.0)),
            })
    elif cfg.get("keywords"):
        for kw, f in cfg["keywords"].items():
            if kw and f:
                out.append({"keywords": [kw], "file": f, "volume": 1.0})
    return out


# ================================================================
# キーワード検出
# ================================================================

def find_hit(text: str, mappings: list[dict],
             last_fire: dict[str, float], cooldown_sec: float, now: float):
    """
    text にマッチする最初のマッピングを返す（クールダウン中のものはスキップ）。
    戻り値: (file, volume) または None。再生時は last_fire を更新。
    """
    for m in mappings:
        if any(kw in text for kw in m["keywords"]):
            f = m["file"]
            if now - last_fire.get(f, 0.0) < cooldown_sec:
                continue
            last_fire[f] = now
            return (f, m["volume"])
    return None


# ================================================================
# 幻覚（ハルシネーション）対策
# ================================================================

def speech_metrics(audio: np.ndarray, frame_sec: float = 0.03):
    """30ms フレーム単位の有音割合と最大音量を返す。"""
    n = int(WHISPER_SR * frame_sec)
    if len(audio) < n:
        return 0.0, 0.0
    usable = audio[: len(audio) // n * n].reshape(-1, n).astype(np.float64)
    fr_rms = np.sqrt((usable ** 2).mean(axis=1))
    voiced_ratio = float((fr_rms > SILENCE_THRESH).mean())
    peak = float(fr_rms.max())
    return voiced_ratio, peak


def has_real_speech(audio: np.ndarray) -> bool:
    """実際の発話が含まれているか（無音・ノイズ片に対する幻覚を弾く）。"""
    ratio, peak = speech_metrics(audio)
    return ratio >= MIN_SPEECH_RATIO and peak >= SPEECH_PEAK_MIN


def _normalize_text(text: str) -> str:
    return re.sub(r"[、。・！？!?,.\s]", "", text)


def is_hallucination_text(text: str) -> bool:
    """既知の幻覚定型句と完全一致（記号除去後）するか。"""
    return _normalize_text(text) in HALLUCINATION_PHRASES


def add_ignore_phrases(phrases) -> None:
    """config の "ignore_phrases" からユーザー定義の除外句を追加する。"""
    for p in phrases or []:
        norm = _normalize_text(p)
        if norm:
            HALLUCINATION_PHRASES.add(norm)


# ================================================================
# 自己ループ防止ゲート
# ================================================================

class Gate:
    """
    効果音の再生中＋直後の一定時間、音声検出を停止するためのゲート。

    効果音の多くは実際の音声（例:「ブロリーです」）なので、再生音が
    スピーカー→ループバック（or マイク）経由で再び拾われ、同じキーワードを
    再検出して無限ループする。それを防ぐため、再生中はキャプチャ側で音声を破棄する。
    """
    def __init__(self):
        self._playing = False
        self._until = 0.0

    def begin(self):
        self._playing = True

    def end(self, tail: float = PLAYBACK_TAIL_SEC):
        self._playing = False
        self._until = time.monotonic() + tail

    def suppressed(self) -> bool:
        return self._playing or time.monotonic() < self._until


# ================================================================
# 音声ユーティリティ
# ================================================================

def rms(a: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


def to_mono_f32(raw: bytes, channels: int) -> np.ndarray:
    """paFloat32 バイト列 → float32 モノラル。"""
    arr = np.frombuffer(raw, dtype=np.float32)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == WHISPER_SR:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(orig_sr, WHISPER_SR)
    return resample_poly(audio, WHISPER_SR // g, orig_sr // g).astype(np.float32)


# ================================================================
# WASAPI ループバックデバイス
# ================================================================

def find_loopback_devices(p: pyaudio.PyAudio) -> list[dict]:
    return list(p.get_loopback_device_info_generator())


def default_output_name(p: pyaudio.PyAudio) -> str:
    """既定の WASAPI 出力デバイス名（ループバックの自動推奨に使用）。"""
    try:
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if info["type"] == pyaudio.paWASAPI:
                return p.get_device_info_by_index(info["defaultOutputDevice"])["name"]
    except Exception:
        pass
    return ""

# ミーム検出ツール (Meme Detector)

Discord などの通話をリアルタイムに文字起こしし、登録したキーワードを検出すると効果音を再生するツールです。

```
[23:10:02] 太郎: それは草
         🔊 sounds/kusa.wav (vol 1.0)（全員に再生）
[23:10:05] 自分: おめでとう！
         🔊 sounds/fanfare.wav (vol 1.0)（全員に再生）
```

## 仕組み

Discord のボイスチャンネルは E2EE（DAVE プロトコル）で暗号化されており、bot が受信した音声を復号する方法は現状安定していません。このツールは発想を変え、**自分の PC が再生している音声（＝復号済みの全員の声）を WASAPI ループバックで録音**して文字起こしします。

```
スピーカー出力（相手の声）──┐
                            ├─→ 録音 → faster-whisper → キーワード検出 → 効果音
マイク入力（自分の声）──────┘
```

- 音声の取りこぼしなし（E2EE・bot の制約を受けない）
- bot は「効果音の再生」と「誰が喋っているかの検出」のみに使用（音声は受信しない）

## 2つの動作モード

| | `detector.py`（ローカル版） | `meme_bot.py`（bot再生版） |
|---|---|---|
| 効果音を聞く人 | 自分だけ | **通話相手全員** |
| 音声認識の対象 | 相手（ループバック）のみ | 相手＋自分（マイク） |
| 話者の表示 | なし | 個別の話者名（`太郎:` など） |
| 必要なもの | なし | Discord Bot Token, ffmpeg |

## ファイル構成

```
meme-detector/
├── core.py              # 共有ロジック（設定・キーワード検出・音声処理・幻覚対策）
├── detector.py          # ローカル版エントリポイント
├── meme_bot.py          # bot再生版エントリポイント
├── config.example.json  # 設定テンプレート（コピーして config.json を作る）
├── requirements.txt
└── sounds/              # 効果音（wav/mp3/ogg/flac）
```

## セットアップ

1. 依存パッケージをインストール（Python 3.10+ / Windows）

   ```
   pip install -r requirements.txt
   ```

2. [ffmpeg](https://ffmpeg.org/download.html) をインストールして PATH を通す（bot再生版のみ）

   > NVIDIA GPU があれば自動で使われ、`medium`/`large-v3` でも高速・高精度です（`device: "auto"`）。CUDA/cuDNN ランタイムが必要な環境では別途用意してください。

3. 設定ファイルを作成

   ```
   copy config.example.json config.json
   ```

4. **bot再生版を使う場合**: [Discord Developer Portal](https://discord.com/developers/applications) で Bot を作成
   - `Privileged Gateway Intents` で **SERVER MEMBERS** と **MESSAGE CONTENT** を ON
   - Token を `config.json` の `"token"` に貼り付け（**`config.json` は絶対に公開しないこと**。`.gitignore` 済み）
   - Bot をサーバーに招待（`bot` スコープ、接続・発言などの音声権限）

## 使い方

### ローカル版

```
python detector.py
```

初回はループバックデバイスの選択メニューが出ます（★推奨＝既定の出力を Enter で選択）。選択は config.json に保存され、次回から自動です。

### bot再生版

```
python meme_bot.py
```

起動後、Discord のボイスチャンネルに入ってからテキストチャンネルで:

| コマンド | 動作 |
|---|---|
| `!join` | 自分のいる VC に bot を呼び、検出開始 |
| `!leave` | 退出・停止 |
| `!status` | 状態確認 |
| `!who on` / `!who off` | 話者名表示 ⇔ 自分/相手のシンプル表示 |
| `!kw list` | 効果音マッピング一覧 |
| `!kw add <ファイル> <単語...>` | キーワード追加（例: `!kw add sounds/laugh.mp3 笑 わら`） |
| `!kw del <単語>` | キーワード削除 |

## 設定リファレンス (config.json)

| キー | 説明 | 既定値 |
|---|---|---|
| `token` | Bot Token（bot再生版のみ） | — |
| `model` | Whisper モデル。tiny/base/small/medium/large-v3 | `small` |
| `language` | 認識言語 | `ja` |
| `device` | `auto`/`cpu`/`cuda`。auto は GPU があれば自動で使用 | `auto` |
| `compute_type` | `auto`/`int8`/`float16` 等（auto: GPUなら float16, CPUなら int8） | `auto` |
| `loopback_device` | 相手の声の録音元デバイスID（null=自動/選択） | `null` |
| `mic_device` | 自分の声のマイクID（null=既定のマイク） | `null` |
| `output_device` | 効果音の再生先（ローカル版のみ） | `null` |
| `enable_loopback` / `enable_mic` | 相手/自分の声を認識するか | `true` |
| `identify_speakers` | 相手を個別の話者で表示（bot再生版） | `true` |
| `anonymize_names` | `true` で実名を匿名通称（太郎・次郎…）に置き換え | `false` |
| `self_user_id` | 自分の Discord ユーザーID。誤割当防止（通常は不要: `!join` した人を自動除外） | `null` |
| `silence_sec` | 無音がこの秒数続いたら文字起こし。小さいほど低遅延だが短く切れて精度が下がる | `0.8` |
| `max_buf_sec` | 長い発話を区切る上限秒 | `6.0` |
| `beam_size` | Whisper のビーム幅。大きいほど高精度・低速（`1`で最速、`5`で高精度） | `5` |
| `cpu_threads` | 文字起こしの CPU スレッド数（null=自動） | `null` |
| `workers` | 文字起こしの並列数。同時発話で詰まらないように | `2` |
| `cooldown_ms` | 同じ効果音が連続で鳴るのを抑制するミリ秒 | `2500` |
| `ignore_phrases` | 誤認識として無視する定型句のリスト | `[]` |
| `mappings` | キーワード→効果音の対応（下記） | — |

### 効果音マッピング

```json
{
  "keywords": ["なにこれ", "何これ", "なにそれ"],  // どれか1つでも含まれたら鳴る
  "file": "sounds/nanikore.mp3",
  "volume": 0.5                                     // 0.0〜1.0
}
```

- 部分一致（発話の文章中に含まれていれば鳴る）
- 1発話につき最初にマッチした1件のみ再生。同じ効果音は `cooldown_ms` の間は再生されない
- 効果音は `sounds/` に置く。WAV/MP3/OGG/FLAC 対応

## 工夫している点（実装メモ）

- **自己ループ防止**: 効果音の多くは音声（セリフ）なので、再生音を自分で拾って無限ループしないよう、再生中＋直後1秒は検出を停止する（`core.Gate`）
- **幻覚対策**: Whisper は無音・ノイズに「ご視聴ありがとうございました」等を出力する。①音声エネルギーが低い区間は文字起こししない ②既知の幻覚句を除外、の2段で抑制
- **個別話者の特定**: bot は音声を復号できないが、音声パケットの到着タイミングから「誰がいつ喋っているか」は分かる（py-cord の SpeakingTimer、0.2秒途切れで発話終了と判定）。これとループバック音声の発話区間を時間で突き合わせて話者名を割り当てる
  - `!join` した本人と bot は割当から自動除外（ループバックに本人の声は入らないため）
  - 話者が入れ替わったら無音を待たずにバッファを区切る（会話のテンポに追従）
  - 判定が拮抗したら誤った名前を出さず `相手:` にフォールバック
- **並列ワーカー**: 複数人が同時に話しても文字起こしが詰まらないよう並列処理。処理が追いつかなければ古い音声を破棄して最新を優先

## 精度と速度の調整

精度（聞き取りの正確さ）と速度（反応の速さ）はトレードオフです。`config.json` で調整します。

### GPU があるなら最優先（精度・速度を両立）

`device: "auto"`（既定）なら **NVIDIA GPU を自動で使用**します。GPU だと大きいモデルでも非常に速いので、`model` を `medium` や `large-v3` にするのがおすすめ（精度が大きく上がり、かつ高速）。

参考実測（RTX 4070 SUPER, beam=5, 3秒の音声）:

| 構成 | 処理時間 |
|---|---|
| GPU `large-v3`（最高精度） | 約 240 ms |
| GPU `medium` | 約 150 ms |
| CPU `small` | 約 1400 ms |

### CPU のみの場合（精度と速度のトレードオフ）

| 重視するもの | 設定の目安 |
|---|---|
| **精度重視** | `beam_size: 5` / `silence_sec: 0.8` / `model: "small"` |
| **速度重視** | `beam_size: 1` / `silence_sec: 0.5` / `model: "base"` |

- `beam_size`: Whisper の探索幅。`5`（高精度・既定）↔ `1`（最速）
- `silence_sec`: 小さいと反応は速いが、発話が途中で切れて精度が落ちる
- `model`: `tiny < base < small < medium < large-v3`（右ほど高精度・低速）

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| 認識が遅い | `model` を `base` に、`beam_size` を `1` に、`silence_sec` を `0.5` に |
| 誤認識が多い・聞き取りが甘い | `beam_size` を `5` に（高精度）、`silence_sec` を `0.8`〜`1.0` に、`model` を `medium` に |
| 「ご視聴ありがとう」等が出る | 既知の Whisper 幻覚。自動除外済み。新しい幻覚句は `ignore_phrases` へ |
| 効果音でループする | 自動防止済み。残響が強い環境は `core.py` の `PLAYBACK_TAIL_SEC` を 1.5〜2.0 に |
| 話者名がよく間違う | `!who off` で自分/相手表示に切替 |
| ボイス接続が 4017 で切れる | `davey` がインストールされているか確認（`pip install davey`） |

## 配布（EXE化）

```
pip install pyinstaller
pyinstaller --onefile --add-data "sounds;sounds" detector.py
```

Whisper モデルは初回起動時に自動ダウンロードされます（要ネット接続）。

## 注意事項

- **Bot Token は絶対に公開しない**こと（`config.json` は `.gitignore` で除外済み）
- 通話を録音・文字起こしするツールなので、**参加者に伝えたうえで**使うことを推奨します
- 効果音の音源は各自で用意し、権利にご注意ください（`.gitignore` の既定では mp3 はリポジトリに含まれません）
- Windows 専用です（WASAPI ループバックを使用）

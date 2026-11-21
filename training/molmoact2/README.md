# MolmoAct2-Think fine-tuning 手順書（湿布貼付タスク）

キュレーション済みの OpenArm WebXR teleop データセットで
[`allenai/MolmoAct2-Think`](https://huggingface.co/allenai/MolmoAct2-Think) を fine-tuning する手順です。

主な変更点は次の 3 つです。

- **深度推論の教師に ZED の実測深度を使う。** 通常 MolmoAct2 は Depth-Anything の推定深度を使います。
- **湿布を貼る位置の bbox を front 画像（`head_left`）に描き込んで入力する。** アノテーションは左カメラに対して付いています。
- **bbox-action binding 損失を加える。** ListenVLA（[arXiv:2610.16313](https://arxiv.org/abs/2610.16313)、Sec. III-C）を移植したもので、bbox を変えたときに行動が追従するよう学習します。

| 項目 | 設定 |
| --- | --- |
| 入力カメラ | `head_left`（bbox あり）、`wrist_right`。`head_right` と `wrist_left` は使わない |
| state / action | 両腕 7 関節 + グリッパの 16 次元、関節角の絶対値 |
| control_mode | `absolute joint pose`（事前学習データの大半と同じ表現） |
| setup_type | `bimanual openarm robotic arms` |
| action horizon | 30 ステップ（30 fps で 1 秒） |
| 開始チェックポイント | `allenai/MolmoAct2-Think` |
| 学習コード | `lib/molmoact2/experiments`（公式リリース版）。編集はせず、`molmoact2_patches.py` で実行時に挙動を変える |

> **注意:** このコードは構文チェックしかしていません。データ生成も学習も未実行です。
> 必ず手順 4 の smoke テストを通してから本番の学習を始めてください。

---

## 0. 前提

| 項目 | 内容 |
| --- | --- |
| GPU | NVIDIA（CUDA 12.8 対応ドライバ）。LoRA は H200 1 枚、フル FT は H200 複数枚を想定。smoke テストは 24GB でも動く想定 |
| ディスク | モデル約 22GB（`~/.cache/huggingface`）、データセット約 15GB、チェックポイント。数十 GB の空きが必要 |
| ツール | `git`、[`uv`](https://docs.astral.sh/uv/)、Python 3.12（uv が取得する） |
| ネットワーク | 初回実行時に Hugging Face から次をダウンロードする（認証不要）: `allenai/MolmoAct2-Think`（約 22GB）、深度用の VQ-VAE と Depth-Anything（`allenai/MolmoAct-7B-D-0812`） |

## 1. リポジトリとサブモジュールの取得

```bash
git clone https://github.com/Azuma413/dora-openarm-data-collection.git
cd dora-openarm-data-collection
git submodule update --init lib/molmoact2   # 学習に必要なのはこれだけ
```

`lib/molmoact2` の中のサブモジュール（`lerobot`、`YAM`、`EVA_DROID`）は初期化しなくて構いません。
学習コードは `lib/molmoact2/experiments/` に LeRobot ごと同梱されています。

## 2. 環境構築

```bash
cd training/molmoact2
./setup_env.sh
```

`training/molmoact2/.venv` に次の環境を作ります: torch 2.10（cu128）、`ai2-molmo`（experiments）、同梱の LeRobot 0.5.0。

- 同梱 LeRobot の依存 `pynput → evdev` はカーネルヘッダがないとビルドできず、学習にも使いません。そのため LeRobot は `--no-deps` で入れ、必要な依存だけを個別に入れています。
- CUDA のバージョンを変える場合は `CUDA_VERSION_PATH=cu126 ./setup_env.sh` のように指定します。

動作確認:

```bash
.venv/bin/python -c "import torch, olmo, lerobot, peft; print(torch.__version__, torch.cuda.is_available())"
```

## 3. データの準備

### 3.1 元データセットを置く

キュレーション済みデータセット（LeRobot v3.0）をサーバに置きます。

```
~/openarm-datasets/webxr-pedestal-zed-lerobot-curated/
├── data/chunk-000/*.parquet      # state, action, observation.images.head_depth（ZED 深度, 16bit TIFF, mm）
├── videos/observation.images.{head_left,head_right,wrist_left,wrist_right}/
└── meta/{info.json, stats.json, tasks.parquet, episodes/, target_bbox_frames.csv, ...}
```

使うのは `meta/target_bbox_frames.csv`（全フレーム分の bbox。座標は `head_left` の 1280×720 px）です。

### 3.2 学習用データを生成する

```bash
./prepare_data.sh ~/openarm-datasets/webxr-pedestal-zed-lerobot-curated
```

内部では次の 4 段階が順に走ります。GPU を使います。

| 段階 | スクリプト | 出力 |
| --- | --- | --- |
| ① 学習用ビュー | `prepare_dataset.py` | `~/openarm-datasets/molmoact2/<repo>-molmoact2/` |
| ② 深度の比較 | `generate_zed_depth_annotation.py --compare-depth-anything 12` | `~/openarm-datasets/molmoact2-depth/<repo>-molmoact2-compare/` |
| ③ 深度トークン | `generate_zed_depth_annotation.py` | `~/openarm-datasets/molmoact2-depth/<repo>-molmoact2/` |
| ④ binding ペア | `mine_binding_pairs.py` | `<学習用ビュー>/meta/binding_pairs.parquet` |

`<repo>` は元データセットのディレクトリ名です。

各段階の詳細:

- **① 学習用ビュー**
  - 使うカメラだけを残し、`target_bbox` 列を追加し、深度画像の列を落とします。
  - 動画は元データセットへのシンボリックリンクなので、**元データセットは移動・削除しないでください**。
- **② 深度の比較**
  - サンプルしたフレームで、ZED と Depth-Anything を比較します。
  - 出力は `summary.json` と比較画像です。画像は左から「RGB | ZED | Depth-Anything | ZED コードの VQ-VAE 再構成」の順に並びます。
- **③ 深度トークン**
  - ZED 深度を逆深度にし、フレームごとに正規化し、穴を埋めてから VQ-VAE で 10×10 のコードにします。
  - 前フレームからの更新判定は RGB の変化で行います。
- **④ binding ペア**
  - ListenVLA の「文脈一致ペア」探索です。同じタスクの別エピソードから、状態と画像が近く、bbox と行動が異なるフレームを探します。

保存先は環境変数で変えられます: `LEROBOT_DATA_ROOT`、`LEROBOT_DEPTH_DATA_ROOT`、`OPENARM_REPO_ID`。
変えた場合は、学習時（手順 4・5）にも同じ値を設定してください。

### 3.3 生成結果を確認する

**深度（②の `summary.json`）**

| 指標 | 意味 | 目安 |
| --- | --- | --- |
| `spearman_zed_vs_da` | ZED と Depth-Anything の深度マップの順位相関 | 高いほど VQ-VAE の学習分布に近い |
| `recon_mae_zed` / `recon_mae_da` | 各深度の VQ-VAE 再構成誤差 | ZED が DA より大きく悪化していないこと |
| `zed_invalid_frac` | ZED の欠損画素の割合 | 大きすぎる場合は `--min-mm` / `--max-mm` を見直す |

比較画像で、ZED の深度画像が近い物体ほど明るくなっていること、背中の形状が読み取れることを目で確認してください。
正規化の挙動は `--lo-pct` / `--hi-pct`（既定 1 / 99 パーセンタイル）で調整できます。

**binding ペア（④の標準出力）**

- `eligible_anchors` と `anchors_with_pair` を確認します。全フレームのうちペアが見つかったフレームの数です。
- 少なすぎる場合（数百以下）: `--joint-tol`、`--history-tol`、`--fingerprint-tol` を緩めます。
- 多すぎて行動差が小さい場合: `--min-action-diff` を上げます。
- 再実行は `.venv/bin/python mine_binding_pairs.py ~/openarm-datasets/molmoact2/<repo>-molmoact2 [オプション]` です。

既定の閾値:

| 条件 | 既定値 |
| --- | --- |
| 関節角の差 | 各 0.05 rad 以内 |
| グリッパの差 | 0.004 以内 |
| 直前 4 ステップの動きの差 | RMS 0.02 rad 以内、静止/移動の状態も一致 |
| 画像（bbox なし）の 8×8 縮小の差 | RMS 0.20 以内 |
| bbox 中心の距離 | 20 px 以上 |
| 先頭 8 ステップの正規化行動の差 | 0.02 以上 |

## 4. smoke テスト（必須）

```bash
PROFILE=smoke ./run_train.sh
```

LoRA、バッチ 1 で 10 ステップ回します。次の点を確認してください。

- データ読み込み・前処理・forward/backward がエラーなく通ること
- ログに `train/binding/*` が出ていること。出ない場合はペアが読めていません
- `action_flow_loss` と CE 損失（深度トークン）が有限値であること

最初の実行ではモデル（約 22GB）のダウンロードが走ります。

## 5. 本番の学習

```bash
# LoRA（小規模データ向け、README の推奨）
PROFILE=h200 NPROC=1 ./run_train.sh

# フル FT
PROFILE=h200 NPROC=8 MODE=fft ./run_train.sh

# 追加の trainer フラグは末尾に付ける
PROFILE=h200 NPROC=1 ./run_train.sh --max_duration=5000
```

### `PROFILE=h200` の既定値

| 項目 | 既定 | 変更方法 |
| --- | --- | --- |
| device batch / global batch | 8 / 64 | `DEVICE_BATCH`、`GLOBAL_BATCH` |
| ステップ数 | 15000 | `MAX_STEPS` |
| 保存間隔 | 2500 ステップ（直近 4 つを保持） | `SAVE_INTERVAL` |
| 保存先 | `training/molmoact2/checkpoints/<EXP_NAME>/` | `SAVE_FOLDER`、`EXP_NAME` |

### `MODE` ごとの学習率

| MODE | LLM | ViT | connector | action expert |
| --- | --- | --- | --- | --- |
| `lora`（rank 64, dropout 0） | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| `fft` | 1e-5 | 5e-6 | 5e-6 | 5e-5 |

LoRA の dropout を 0 にしているのは、binding の 2 回の forward で dropout マスクが異なるのを避けるためです。

### 共通の設定（公式の Depth FT レシピに準拠）

`--enable_depth_reasoning=true --num_depth_tokens=128 --num_depth_tokens_per_image=100 --depth_code_input_noise_rate=0.1 --style_robot_action=1.0 --style_robot_depth_action=1.0 --num_flow_timesteps=8 --packing=false --dynamic_seq_len=true`

### ログ

`WANDB_PROJECT`（必要なら `WANDB_ENTITY`）を設定すると wandb に記録します。未設定ならコンソールのみです。

| 指標 | 意味 |
| --- | --- |
| `train/binding/loss` | binding 損失 |
| `train/binding/cosine` | bbox を入れ替えたときの速度変化と、デモの行動差のコサイン類似度。上がっていけば bbox に追従している |
| `train/binding/response_abs` / `target_abs` | 上記の速度変化と行動差の大きさ |
| `train/binding/pairs` | 1 ステップあたりのペア数 |

### binding 関連の環境変数

| 変数 | 既定 | 内容 |
| --- | --- | --- |
| `MOLMOACT2_BINDING_WEIGHT` | 0.5 | binding 損失の重み λ。0 で無効 |
| `MOLMOACT2_BINDING_PROB` | 1.0 | ペアがあるフレームでペアを使う確率 |
| `MOLMOACT2_BINDING_PAIRS` | `<学習用ビュー>/meta/binding_pairs.parquet` | ペアファイルのパス |
| `MOLMOACT2_BBOX_COLOR` | `0,255,0` | bbox の色（RGB） |
| `MOLMOACT2_BBOX_THICKNESS` | 4 | bbox の線幅（1280 px 幅換算） |

## 6. 学習結果

LoRA の場合は `stepN-merged/` が推論用の完全なモデルです（アダプタをマージ済み）。
`stepN/` は学習再開用です。

Hugging Face 形式への変換:

```bash
cd ../../lib/molmoact2/experiments
../../../training/molmoact2/.venv/bin/python -m olmo.hf_model.convert_molmoact2_to_hf \
  ../../../training/molmoact2/checkpoints/<run>/stepN-merged \
  ../../../training/molmoact2/checkpoints/<run>/stepN-hf
```

同梱 LeRobot で読み込む場合は `--policy.type=molmoact2 --policy.checkpoint_path=.../stepN-merged` を指定します。

**推論時は、学習時と同じスタイルで front 画像に bbox を描いてください。**
既定は緑、1280 px 幅で 4 px です。描画は `molmoact2_patches.draw_bbox` と同じ処理にしてください。

---

## 実装メモ

### `molmoact2_patches.py` がやっていること

1. **bbox の描画**
   - `head_left` の元解像度の画像に bbox を描いてから、縮小・augmentation に回します。
   - spatial augmentation をかけても、bbox は画像と一緒に動きます。
2. **深度教師の修正**
   - 公式の wrapper は `--action_format=continuous`（既定値で、README の Depth FT レシピもこれ）のとき、深度トークン列を系列から落としてしまいます。
   - そのままでは深度トークンが teacher forcing されず、CE 損失も付きません。パッチで応答として戻しています。
   - action expert の参照範囲から末尾の EOS も外しています。推論時は深度スパンの直後に action expert が走り、EOS がないためです。
3. **binding 損失**（ListenVLA の式(4)〜(6)。音声の入れ替えを bbox の入れ替えに置き換え）
   - フレーム i にペア j があるとき、i の生画像に j の bbox を描いた画像を作ります。
   - 元の画像と同じ乱数状態で前処理するので、2 つの違いは描いた bbox だけになります。
   - 通常の forward/backward が終わった後に 2 つを流れ時刻 t=0、共通ノイズで推論し、損失 `w_ij · mean[(v(o_i, bbox_i) − v(o_i, bbox_j)) − (A_i − A_j)]²` を計算します。対象は先頭 8 ステップ、正規化した行動です。
   - 重み `w_ij = exp(−D) · clip(bbox中心距離 / 80px, 0, 1)` は勾配を流しません。
   - micro-batch の全行で平均し、ペアのない行の寄与は 0 です。
   - 全体の損失は `L_flow + L_depth + λ·L_bind`（λ=0.5 は ListenVLA 式(6)の 2:1 に相当）です。
   - binding の backward は別に実行するので、ピークメモリは増えません。

### 実行時の注意

- 公式の trainer は DataLoader のワーカーを `spawn` で起動します。そのため、パッチは `site/sitecustomize.py` から全プロセスに当てています。
- 有効化の条件は `PYTHONPATH` に `site/` が入っていて、`OPENARM_MOLMOACT2_PATCH=1` が設定されていることです。`run_train.sh` が両方を設定します。
- `train.py` を直接起動する場合は、この 2 つを自分で設定してください。

### ファイル一覧

| ファイル | 役割 |
| --- | --- |
| `setup_env.sh` | 環境構築 |
| `prepare_data.sh` | データ準備（下の 3 スクリプトを順に実行） |
| `prepare_dataset.py` | 学習用ビューの生成 |
| `generate_zed_depth_annotation.py` | ZED 深度 → 深度トークン、Depth-Anything との比較 |
| `mine_binding_pairs.py` | binding ペアの探索 |
| `molmoact2_patches.py` | 学習コードへのパッチ |
| `train.py` | データセットの登録とパッチの適用、公式 `train_lerobot.py` の起動 |
| `run_train.sh` | 学習の起動（PROFILE / MODE） |
| `site/sitecustomize.py` | spawn ワーカーへのパッチ適用 |
| `common.py` | 共通ヘルパ |

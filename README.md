# OpenArm データ収集（dora-rs）

[OpenArm](https://openarm.dev/) を [dora-rs](https://dora-rs.ai/) で動かし、Meta Quest 3（WebXR）で遠隔操作してデータを収集する手順です。

[enactic/dora-openarm-data-collection](https://github.com/enactic/dora-openarm-data-collection) のフォークです。
[guchio123 のフォーク](https://github.com/guchio123/dora-openarm-data-collection)の運用（この README の手順、
タスクのメタデータ、UI、pedestal/ZED、USB 接続）を引き継ぎ、WebXR テレオペを作り直しています。

- **ラグの原因だった IK の 1 ステップあたりの誤差制限を外しました。** 0.2 m/s・90°/s の手の動きで、
  手首の追従誤差は 23° → 0.8°、止めてから追いつくまで 525 ms → 6 ms（MuJoCo での実測、[docs/teleop.md](docs/teleop.md#measured)）。
- **グリップを握っている間だけ腕が手の動きに追従します（クラッチ）。** 離せば腕はその場で止まるので、手を楽な位置に戻して握り直せます。
- **速度・関節範囲・追従を監視する安全ゲート**が、IK とロボットの間に入っています。
- ゴーグル内に状態表示（HUD）が出ます。B は長押しでのみ終了します。接続にはペアリングトークンが必要です。

仕組みと数値の根拠は [docs/teleop.md](docs/teleop.md)（英語）にあります。

- [0. インストール](#0-インストール)
- [1. タスクとメタデータ](#1-タスクとメタデータ)
- [2. 起動手順](#2-起動手順)
- [3. 収集の操作](#3-収集の操作)
- [4. 終了手順](#4-終了手順)
- [5. 収録したデータを確認する](#5-収録したデータを確認する)
- [6. トラブルシューティング](#6-トラブルシューティング)
- [Configurations](#configurations)（英語）

## 0. インストール

```bash
sudo apt install libcli11-dev coturn   # coturn は USB 接続のときだけ必要
git clone --recursive https://github.com/Raptor-zip/dora-openarm-data-collection.git
cd dora-openarm-data-collection
uv sync
uv run scripts/webxr-setup             # 証明書とペアリングトークン（tls/）。最初の 1 回だけ
```

`scripts/webxr-setup` が表示する `https://…:8443/#k=…` が Quest で開く URL です。`#k=` 以降がトークンで、
これを知っている端末だけがロボットを動かせます。**他人に見せないでください。** 漏れたら
`uv run scripts/webxr-setup --rotate-token` で作り直します（保存済みの URL はすべて無効になります）。

## 1. タスクとメタデータ

タスク名、場所、オペレーターなどのメタデータは、すべて [`metadata.yaml`](metadata.yaml) の **1 ファイル**にまとめて書きます。

```yaml
location: Lab
operator: Tester

tasks:
  - prompt: Pick a spoon.          # UI に表示されるタスク名
    description: Pick a spoon on a table.
  - prompt: Pick a cup.
    description: Pick a cup on a table.
```

## 2. 起動手順

### 2-1. CAN を立ち上げる

```bash
for i in can0 can1; do
  sudo ip link set $i down
  sudo ip link set $i type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set $i txqueuelen 1000
  sudo ip link set $i up
done
ip -br link | grep can     # can0 と can1 が UP になっていれば OK
```

### 2-2. ターミナル 1：USB 中継を起動する

Quest を USB-C でつないだ状態で、次を実行します。

```bash
cd ~/dora-openarm-data-collection
scripts/webxr-usb/start.sh
```

### 2-3. ターミナル 2：dataflow を起動する

```bash
cd ~/dora-openarm-data-collection
source .venv/bin/activate
TASK="Pick a spoon." LEADER=webxr ROBOT=pedestal-zed scripts/dora-envsubst run dataflow.yaml --uv
```

- PC のブラウザで http://localhost:8000/ を開くと、データ収集 UI が表示されます。

起動前に、テレオペのノードどうしの設定（タイミング、周期、速度上限）の食い違いが自動で検査されます。
食い違いがあると起動せず、どこが違うかを表示します（[docs/teleop.md](docs/teleop.md#checks)）。

実機で机の高さを使う場合は、測った値を `TABLE_Z=-0.30`（m、`arm_origin` 基準）のように付けて起動します。
付けないと、実機では机への押し付け防止（指先を机より上に保つ制限）が**無効**になります（起動時に表示されます）。

### 2-4. Quest でテレオペを開始する

1. Quest のブラウザで、`scripts/webxr-setup` が表示した **`https://localhost:8443/#k=…`** を開きます。
   - 証明書の警告が出たら、「詳細設定」→「localhost にアクセスする」で進みます。
2. 数秒待ってから、**Start** を **1 回だけ**押します。
3. **両手のトリガーを 1 回ずつ引きます。** フォロワーが指令を受け付け始め、UI の Arm 表示が `ALIGNED` になります
   （Start Arm のたびに 1 回必要です）。腕は今いる位置で止まったままです。
4. **グリップ（中指）を握ると**、その腕が手の動きに追従します。離すとその場で止まります。
   - 握る前に、手を 0.3 秒ほど止めてください。動かしながら握っても追従は始まりません。
   - ゴーグル内の表示が `R:FOLLOW` になっていれば右腕が追従中です。

## 3. 収集の操作

### コントローラー

| 操作 | 動作 |
|---|---|
| **グリップ**（中指）を握ったまま手を動かす | その腕の手先が、握った時点からの手の動きの分だけ動く |
| グリップを離す | その腕はその場で止まる。手を楽な位置に戻して、握り直せばまた動く |
| **トリガー**（人差し指） | グリッパーを開閉する（引くと閉じる） |
| **A ボタン**（右手） | 待機中：エピソードを**開始**<br>収録中：**成功**として保存 |
| **B ボタン**（右手） | 収録中：**失敗**として保存<br>待機中：**2 秒長押し**で終了（dataflow を止める） |

頭を動かしたり体を傾けたりしても、腕は動きません。
コントローラーを見失った手は、その場で一時停止します。0.5 秒以内に同じ位置へ戻れば、握り直さずに再開します。

### ゴーグル内の表示（HUD）

| 表示 | 意味 |
|---|---|
| `READY` / `REC 01:05  ep 7` | 待機中 / 収録中（経過時間、エピソード番号） |
| `R:FOLLOW` | 右腕が手に追従している |
| `R:HOLD` | グリップを離していて、腕は止まっている |
| `R:REGRIP` | いったんグリップを離してから握り直す必要がある（起動直後、Start Arm の後など） |
| `R:LATE` | コントローラーの入力が途切れている |
| `GATE R:HOLDING(理由)` | 安全ゲートが右腕を止めている。理由は [docs/teleop.md](docs/teleop.md#gate-dora-openarm-gate) |

開始・成功・失敗で右手、取り消しで左手、終了で両手のコントローラーが振動します。

### PC の UI（http://localhost:8000/）

| ボタン | キー | 動作 |
|---|---|---|
| Start! | `c` | エピソードを開始する |
| Success! | `c`（収録中） | 成功として保存する |
| Fail! | `a`（収録中） | 失敗として保存する |
| Cancel! | `a` 長押し（収録中） | 保存せずに破棄する |
| Skip task | `c` 長押し | 次のタスクに切り替える |
| Start Arm / Pause Arm | `b` 長押し | 腕の起動 / 停止 |
| Quit | `b` | 終了する |

### 1 エピソードの流れ

1. 腕が `ALIGNED` で、ゴーグル内に `GATE` の行が出ておらず、物体が初期位置に置かれていることを確認します。
2. **A**（または UI の Start!）を押して、収録を開始します。
3. タスクを実行します。
4. 成功したら **A**、失敗したら **B** を押します。撮り直す場合は、UI の Cancel! で破棄します。
5. 物体を初期位置に戻し、1 に戻ります。

> 開始してすぐに A を押すと、0.2 秒ほどのごく短いエピソードが保存されます。押す前に、いま収録中かどうかを UI で確認してください。

### 腕を初期姿勢に戻す

UI で **Pause Arm** を押してから **Start Arm** を押します。

- **Pause Arm**：終了姿勢（肘を 90°）へ 2 秒かけて移動したあと、**モーターの力が抜けます**。重力で腕が落ちることがあるので、**支えられる状態で押してください**。
- **Start Arm**：モーターに力を入れ、開始姿勢へ移動します。そのあと両手のトリガーを 1 回ずつ引き、グリップを握り直すと、開始姿勢から操作を再開できます。

## 4. 終了手順

1. 収録中でないことを確認してから、**B を 2 秒長押し**します（または UI の Quit）。
2. 腕が終了姿勢に移動して止まったら、ターミナル 2 で dataflow が終了したことを確認します。終了しない場合は Ctrl+C で止めます。
3. ターミナル 1 の USB 中継を Ctrl+C で止めます。
4. 腕を支えてから、ロボットの電源を切ります。

## 5. 収録したデータを確認する

```bash
# エピソードの一覧（成功/失敗、長さ、カメラ）
scripts/view-episode.py data/webxr-pedestal-zed/dataset

# エピソード 2 を Rerun ビューアで再生（関節のグラフ、カメラ画像、深度）
scripts/view-episode.py data/webxr-pedestal-zed/dataset 2

# 画面のない環境では .rrd に書き出し、あとで `rerun ep2.rrd` で開く
scripts/view-episode.py data/webxr-pedestal-zed/dataset 2 --save ep2.rrd
```

データの構成は次のとおりです。

```
data/<LEADER>-<ROBOT>/dataset/
├── metadata.yaml                                # エピソード一覧（成功/失敗、task_index）
└── episodes/<番号>/
    ├── obs/arms/{left,right}/state.parquet      # 実際の関節（qpos, qvel, qtorque）
    ├── action/arms/{left,right}/state.parquet   # 指令値
    └── cameras/<カメラ名>/<時刻ns>.jpeg | .png  # 深度は 16bit PNG（mm、0 は深度なし）
```

- UI に表示されるエピソード番号は、起動するたびに 0 から始まります。既存の番号とぶつかった場合は、空いている次の番号で保存されます。実際に保存された番号は、上の一覧で確認してください。
- 不要なエピソード（極端に短いものなど）は、収集を終えたあとで整理してください。

## 6. トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| `uv: command not found` | [uv](https://docs.astral.sh/uv/getting-started/installation/) をインストールします。 |
| Quest でページが開かない / `wrong or missing pairing token` と出て接続できない | URL に `#k=…` が付いているか確認します（`scripts/webxr-setup` が表示したもの）。トークンを作り直した場合は新しい URL を開きます。USB の場合は、`adb devices` が `device` になっているか確認します。`unauthorized` なら、ゴーグル内で USB デバッグを許可します。そのあと `scripts/webxr-usb/start.sh` を起動し直します。 |
| Start を押しても反応しない | 前回の XR セッションが残っています（コンソールに `already an active, immersive XRSession` と出ます）。Quest の**ブラウザアプリを完全に終了**して開き直します。それでも直らなければ、Quest を再起動します。 |
| Start は押せるのに腕が動かない | ターミナル 1 の USB 中継（TURN）が止まっていないか確認します。Wi-Fi では端末どうしの通信が遮断されていることがあるので、USB 接続を使ってください。 |
| 片方の腕が `ALIGNED` にならない | ロボット側のログに `arm[N] went silent` と出ていれば、N 番（0 から数える）のモーターが応答していません。たとえば `arm[5]` は J6（手首の曲げ）です。その付近のケーブルと電源を確認します。 |
| ログに `write_no_buffer` が大量に出る | CAN の送信キューが小さすぎます。`txqueuelen 1000` を設定します（[2-1](#2-1-can-を立ち上げる) を参照）。 |
| グリップを握っても腕が動かない | ゴーグル内の表示を見ます。`REGRIP` ならいったん離して握り直します。`--` や `LATE` ならコントローラーが追跡されていません。`GATE … HOLDING` なら理由を [docs/teleop.md](docs/teleop.md#gate-dora-openarm-gate) で確認します。手を止めずに握ると追従は始まりません。 |
| 起動時に `dora-envsubst: leader: …` と出て起動しない | テレオペのノードどうしの設定が食い違っています。メッセージの項目を [`modules/leader-webxr.yaml`](modules/leader-webxr.yaml) で直します（[docs/teleop.md](docs/teleop.md#checks)）。 |
| 追従が遅い | IK に `--frame-position-error-limit` などの誤差制限を付けていないか確認します（付けると遅くなります。[docs/teleop.md](docs/teleop.md#measured)）。実機では安全ゲートが関節速度をドライバ上限の 0.5 倍に抑えています。 |
| UI のボタンを押しても反応が遅い | UI のページを再読み込みします（改善済みですが、古いページを開いたままの場合に起こります）。 |
| エピソードが保存されていない | レコーダーのログ（`out/<実行ID>/log_recorder.jsonl`）を確認します。`metadata does not match` と出ていれば、[1 章](#1-タスクとメタデータ)の注意を参照してください。 |
| 深度の表示が真っ黒 / ちらつく | `scripts/view-episode.py` を使えば、色の範囲を固定したカラーマップで表示されます。 |

### ログの場所

dataflow のログは、実行ごとに `out/<実行ID>/log_<ノード名>.jsonl` に保存されます。

```bash
d=$(ls -td out/*/ | head -1)       # 最新の実行
tail -5 $d/log_robot.right.jsonl   # 右腕
tail -5 $d/log_recorder.jsonl      # レコーダー
tail -5 $d/log_leader.gate.jsonl   # 安全ゲート（止めた理由）
```

Quest のブラウザ側のエラーは `out/quest.log` に保存されます。

## Configurations

[`dataflow.yaml`](dataflow.yaml) is the only dataflow. The leader and the robot
are modules under [`modules/`](modules) picked by environment variables:

| Variable | Values | Picks | Default |
|---|---|---|---|
| `LEADER` | `ker`, `vr`, `webxr`, `dummy` | `modules/leader-$LEADER.yaml` | `dummy` |
| `ROBOT` | `cell` (real Cell), `pedestal`, `pedestal-zed`, `mujoco`, `dummy` | `modules/robot-$ROBOT.yaml` | `dummy` |

dora doesn't expand environment variables in `module:` paths
([dora-rs/dora#3805](https://github.com/dora-rs/dora/issues/3805)), so run the
dataflow through [`scripts/dora-envsubst`](scripts/dora-envsubst). It expands
`$VAR`, `${VAR}` and `${VAR:-default}` in `module:` paths, writes the result to
`<name>.envsubst.yaml` and runs the given dora command on it.

`ROBOT=cell` and `ROBOT=pedestal-zed` capture the head ZED Mini with the ZED
SDK, which computes depth on the GPU. Install the ZED SDK first; see
[`nodes/dora-openarm-zed`](nodes/dora-openarm-zed/README.md). The depth is
recorded as `cameras/head_depth/<timestamp>.png` (16-bit, mm) and shown in
the UI.

Only the selected leader and robot are in the dataflow, so the others open no
device or port. `LEADER=dummy` with `ROBOT=cell` or `ROBOT=pedestal` moves the
real arms with generated motion. See [`docs/design.md`](docs/design.md) for how
the modules are structured.

With `LEADER=webxr`, `TABLE_Z` (m, in the `arm_origin` frame) is the
measured table for the clutch on a real robot; see [`docs/teleop.md`](docs/teleop.md).
`scripts/dora-envsubst` checks the teleop nodes' couplings before `run`,
`start` and `build` (`SKIP_TELEOP_CHECK=1` skips it).

`TASK` picks the task to start with from the `tasks` in [`metadata.yaml`](metadata.yaml),
by index or by its prompt (exact, or a unique part of it, case-insensitive).
It defaults to the first task, and Skip in the UI still moves to the next one.
An unknown or ambiguous `TASK` stops the UI with the list of tasks:

```bash
TASK="Pick a cup." LEADER=webxr ROBOT=mujoco scripts/dora-envsubst run dataflow.yaml --uv
```

All dataset metadata is in one file, [`metadata.yaml`](metadata.yaml): the
location, the operator, the tasks and the equipment common to every robot,
plus a `robots:` section with what each robot adds, picked by `ROBOT`.
Datasets are recorded under `data/$LEADER-$ROBOT/`, e.g. `data/webxr-mujoco/`:

| Robot | Recorded equipment |
|---|---|
| `cell` | Arms, lifter, five cameras, head depth |
| `pedestal`, `pedestal-viz` | Arms only |
| `pedestal-zed` | Arms, head ZED Mini with depth |
| `mujoco` | Arms, five simulated cameras |
| `dummy` | Arms, five dummy cameras |

The recorder refuses to append to a dataset recorded with different
metadata, e.g. after editing the tasks, so move `data/$LEADER-$ROBOT/` away
first.

### KER leader

`LEADER=ker` uses a [KER](https://github.com/enactic/dora-openarm-ker) leader arm
connected over USB. Its joint positions are sent to the follower arms directly.

### VR leader

`LEADER=vr` receives VR controller poses over UDP by
[dora-openarm-vr](https://github.com/enactic/dora-openarm-vr), converts them to
joint positions by inverse kinematics
([dora-openarm-kinematics](https://github.com/enactic/dora-openarm-kinematics))
and sends them to the follower arms.

### WebXR leader

`LEADER=webxr` uses [dora-openarm-webxr](https://github.com/enactic/dora-openarm-webxr).
It starts a Web server and the Web browser on a VR device such as Meta Quest 3
or PICO 4 connects to it to stream controller poses. No native VR application is
needed.

WebXR requires HTTPS, so a TLS certificate is needed. A self-signed certificate is enough; see the [dora-openarm-webxr setup instructions](https://github.com/enactic/dora-openarm-webxr#setup) for how to generate one. Then run:

```bash
./nodes/dora-openarm-webxr/example/prepare_tls.sh $(hostname).local
LEADER=webxr ROBOT=mujoco scripts/dora-envsubst run dataflow.yaml --uv
```

Open http://localhost:8000/ on the local machine for the data collection UI, and open `https://<host name>:8443/` in the Web browser on your VR device (where `<host name>` is the value passed to `prepare_tls.sh`) to start teleoperation.

### Dummy leader

`LEADER=dummy` generates leader motion without any device. With `ROBOT=dummy`,
we can test the dataflow without any hardware.

### Cell robot

`ROBOT=cell` uses real OpenArm units on CAN, the Cell lifter and wrist, head and
ceiling cameras.

The Cell lifter receives the shared 250 Hz tick and records its elevation action
and observation in millimeters. It uses its built-in startup calibration and hold
behavior, and the left joystick of a VR leader moves it. A real lifter is
required on `can2` by default, and startup enables and moves it for calibration.

### Pedestal robot

`ROBOT=pedestal` uses real OpenArm units on a pedestal over CAN. It has no
lifter and no cameras. The followers use the `openarm_pedestal` configuration.

### MuJoCo robot

`ROBOT=mujoco` uses a MuJoCo simulation
([dora-openarm-mujoco](https://github.com/enactic/dora-openarm-mujoco)) instead
of real OpenArm units and cameras. We can use this for testing teleoperation and
data collection without real hardware.

### Dummy robot

`ROBOT=dummy` doesn't use any hardware. The arms echo their targets and the
cameras send test images.

## Debugging OpenArm libraries

The OpenArm-specific dependency repositories are checked out under [`lib/`](lib/README.md).
Initialize them with `git submodule update --init --recursive`.

Libraries normally come from the package index at the versions in `uv.lock`.
To use an editable library checkout, run the command for the library you need
from the repository root:

```bash
uv add --editable lib/openarm_driver
# Other optional library checkouts:
# uv add --editable lib/openarm_can/python
# uv add --editable lib/openarm_ker
# uv add --editable lib/openarm_control
# uv add --editable lib/openarm_mujoco
```

`uv add --editable` updates the parent dependency declaration, adds a local path
under `[tool.uv.sources]`, updates the lockfile, and syncs the environment.
Restart the dataflow after editing Python code. CAN bindings use
`lib/openarm_can/python`; changes to their C++ code require rebuilding.

To return to the package-index version, remove that library's entry from
`[tool.uv.sources]`, then run `uv lock` and `uv sync --locked`. Keep required
version constraints such as `openarm-driver>=0.5.1`. Avoid separate `uv pip install`
commands for managed dependencies, since they bypass the project lockfile.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

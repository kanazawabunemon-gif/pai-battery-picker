# ゴミに紛れた電池を見つけて抜き取る AIロボットアーム

Physical AI 応用講座1 最終課題。家庭ゴミのトレイに紛れたリチウムイオン電池やモバイルバッテリーを、真上のカメラとAI（YOLO-OBB）で見つけ、SO-101 ロボットアームで抜き取って回収箱へ入れます。物理シミュレータは MuJoCo です。

- 動画：https://youtu.be/seCF616T6F0 （デモ部分は2倍速）
- 説明資料：omnicampus に提出したPDF（説明資料_PAI最終課題_Kanazawabunemon.pdf）

電池が他のゴミの下に隠れていたり、壁際や密集していて指が入らなかったりするときは、ロボット自身が次の行動を選んでから回収します。

1. 上に乗ったゴミをどかす
2. 指の着地点をふさいでいるゴミをどかす
3. 閉じた指先で電池を押さえ、空いている方向へずらす
4. 何も見えなければ、背の高いゴミをどかして下を確かめる

## 結果（シミュレーション）

| 項目 | 結果 |
|---|---|
| 検出（検証画像500枚） | mAP50 0.990。電池：適合率0.98・再現率0.92 |
| 半分隠れた電池（見えている割合30〜60%）の発見率 | 91%（300トレイ・613個で集計） |
| 48トレイでの危険物の回収率（AI検出） | **76%**（58/76） |
| 同上（検出だけシミュレータの正解に置き換えた場合） | 79%（60/76） |

評価用のトレイは、学習にも調整にも使っていない乱数で作っています（ゴミ6〜10個＋危険物1〜2個）。

## 構成

```
sim/
  trash_scene.py    場面（トレイ・ゴミ8種・回収箱・カメラ）。毎回ランダムに積み上げる
  ik.py             SO-101 の逆運動学（減衰最小二乗法）
  grasp_planner.py  奥行き画像→高さ地図、指を下ろせる向きの選択、ずらす方向の選択
  pick_expert.py    つかむ／運ぶ／ずらす動作の軌道
  declutter.py      見る→判断→1つ行動 を繰り返すループ（つかむ・どかす・ずらす・掘る）
  assets/so101/     SO-101 モデル（MuJoCo Menagerie, Apache-2.0）。so101_pads.xml は指先パッドを追加した版
perception/
  gen_obb_data.py   学習データの自動生成（色分け画像から向き付きの枠を自動で作る）
  train_yolo.py     YOLO11-OBB の学習
  detector_source.py 検出結果＋奥行き → 3次元の位置・向き・幅
  eval_visibility.py 隠れ具合ごとの発見率の評価
scripts/
  eval_e2e.py       一連の動作の評価（トレイごとに1行のJSON）
  preview_trash.py  場面の確認用の画像を出す
video/
  render_demo.py    デモ映像（メインカメラ＋「AIの目」の小窓＋字幕）
  cards.py, assemble.py  動画のタイトル・説明カードと組み立て
tools/colab/        Google Colab にジョブを投げて結果を受け取る仕組み（ワーカーとPC側の道具）
models/yolo11s_obb_trash.pt  学習済みの検出モデル
```

## 動かし方

Python 3.12 以上。

```bash
pip install -r requirements.txt
```

場面を確認する（out/preview_trash/sheet.png に画像が出ます）：

```bash
python scripts/preview_trash.py 0 1
```

学習済みモデルで一連の動作を評価する（重いのでGPU付きの環境を推奨）：

```bash
python scripts/eval_e2e.py --mode yolo:models/yolo11s_obb_trash.pt --seeds 1000 1006 --out out/e2e.jsonl
```

デモ映像を作る：

```bash
python video/render_demo.py --seed 1002 --weights models/yolo11s_obb_trash.pt --out out/scene_1002.mp4
```

学習データの生成と学習：

```bash
python perception/gen_obb_data.py --start 0 --n 500 --out out/obb_ds --split train
python perception/gen_obb_data.py --start 100000 --n 500 --out out/obb_ds --split val
python perception/train_yolo.py --data out/obb_ds --epochs 50 --out out/yolo_obb
```

### Google Colab で動かすときの注意

- MuJoCo を GPU で描画するには、最初に `bash tools/colab/colab_egl.sh` を実行してください。NVIDIA の EGL 登録ファイルが無いと、エラーを出さずにソフトウェア描画（数十倍遅い）になります。
- NVIDIA の EGL ではアンチエイリアスが効き、色分け画像や奥行き画像の物体の境目で値が混ざります。このコードでは、色分けと奥行き用のレンダラだけアンチエイリアスを切っています（`sim/trash_scene.py` の `seg_renderer` / `depth_renderer`）。
- 開発時は `tools/colab/` のワーカーで、評価を8並列で回しました（48トレイで約20分、NVIDIA L4）。

## 簡略化している点

- ゴミは形の変わらない物だけです。紙くずは丸めた形の固い物で代用しています。
- 検出・評価とも合成画像のみで、実写画像や実機では検証していません。
- ゴミが電池から外れるときの細かい物理（引っかかり等）は扱っていません。

## ライセンス

このリポジトリのコードは MIT License です（`LICENSE`）。同梱・利用しているもののライセンスは次のとおりです。

- SO-101 モデル（`sim/assets/so101/`）：Apache-2.0（MuJoCo Menagerie、元は TheRobotStudio）。指先パッドの追加のみ変更しています。
- MuJoCo：Apache-2.0
- Ultralytics YOLO：AGPL-3.0。学習済みモデル `models/yolo11s_obb_trash.pt` は Ultralytics で学習したものです。

# -*- coding: utf-8 -*-
"""Write one Colab worker notebook per node (each notebook file gets its own Colab machine)."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = os.environ.get("PAI_COLAB_BASE", os.path.expanduser("~/GoogleDrive/colab_jobs/pai_battery"))
WORKER_FILE = "colab_worker.py"

MD = """# 電池抜き取りロボ — Colab ワーカー（{wid}）

PC が Google ドライブに置いた仕事（シミュレーション・データ作り・学習）を受け取って、この Colab で計算し、結果をドライブへ返します。

**使い方**
1. メニュー「ランタイム → ランタイムのタイプを変更」で **{rt}** を選ぶ。
2. セル1 → セル2 → セル3 → セル4 の順に実行。**セル4 は動かしたままにしてください**（止めるとセッションが切れやすくなります）。
3. 以後の仕事は PC から投入します。セルを貼り直す必要はありません。
4. 終わらせるときは セル5 を実行（今の仕事が終わってから止まります）。その後ランタイムを解放してください。
"""

CELL1 = """#@title 1. 設定とドライブ接続
WORKER_ID = "{wid}"  #@param {{type:"string"}}
LANES = {lanes}  #@param {{type:"integer"}}
from google.colab import drive
drive.mount('/content/drive')
import os, platform, subprocess, time
BASE = '/content/drive/MyDrive/colab_jobs/pai_battery'
assert os.path.isdir(BASE), 'ドライブに colab_jobs/pai_battery が見つかりません'
gpu = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout.strip()
mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
print(f'worker={{WORKER_ID}} cpus={{os.cpu_count()}} ram={{mem_gb:.1f}}GB python={{platform.python_version()}} gpu={{gpu or "なし"}}')
os.makedirs(f'{{BASE}}/workers/{{WORKER_ID}}', exist_ok=True)
stop = f'{{BASE}}/workers/{{WORKER_ID}}/STOP'
if os.path.exists(stop):
    os.remove(stop)
print('===== セル1 完了', time.strftime('%H:%M:%S'), '=====')
"""

CELL2 = """#@title 2. シミュレータの準備（1〜2分）
import subprocess, os, time
if not gpu:
    subprocess.run('apt-get -qq update && apt-get -qq install -y libosmesa6-dev > /dev/null', shell=True, check=True)
    os.environ['MUJOCO_GL'] = 'osmesa'; os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
else:
    os.environ['MUJOCO_GL'] = 'egl'; os.environ['PYOPENGL_PLATFORM'] = 'egl'
subprocess.run('pip -q install mujoco imageio imageio-ffmpeg PyOpenGL', shell=True, check=True)
r = subprocess.run(['python', '-c', 'import mujoco; m = mujoco.MjModel.from_xml_string("<mujoco><worldbody>'
                    '<geom type=\\\\"sphere\\\\" size=\\\\"0.1\\\\"/></worldbody></mujoco>"); r = mujoco.Renderer(m, 64, 64); '
                    'd = mujoco.MjData(m); r.update_scene(d); print("render ok", r.render().mean(), mujoco.__version__)'],
                   capture_output=True, text=True, env=os.environ)
print(r.stdout, r.stderr[-500:])
assert 'render ok' in r.stdout, '描画の準備に失敗しました（上のエラーを PC 側に伝えてください）'
print('===== セル2 完了', time.strftime('%H:%M:%S'), '=====')
"""

CELL3 = """#@title 3. ワーカーを起動（すぐ終わります）
import shutil, subprocess, json, time
shutil.copyfile(f'{{BASE}}/tools/{worker}', '/content/colab_worker.py')  # ドライブ上の古い版を掴まないよう毎回ローカルへ写す
already = subprocess.run("pgrep -f '[c]olab_worker.py' | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
if already not in ('', '0'):
    print('すでにワーカーが動いています。何もしません。')
else:
    logf = open('/content/worker.log', 'a')
    proc = subprocess.Popen(['setsid', 'python', '/content/colab_worker.py', WORKER_ID, str(LANES)],
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ)
    print('started pid', proc.pid)
hbp = f'{{BASE}}/workers/{{WORKER_ID}}/heartbeat.json'
for i in range(24):
    time.sleep(5)
    if os.path.exists(hbp):
        print('心拍を確認:', open(hbp).read()[:200]); break
else:
    print('心拍が出ません。ログ末尾:'); print(open('/content/worker.log').read()[-1500:])
print('===== セル3 完了', time.strftime('%H:%M:%S'), '=====')
"""

CELL4 = """#@title 4. 見張り（動かしたままにしてください）
import json, time, subprocess
hbp = f'{{BASE}}/workers/{{WORKER_ID}}/heartbeat.json'
while True:
    alive = subprocess.run("pgrep -f '[c]olab_worker.py' | wc -l", shell=True, capture_output=True,
                           text=True).stdout.strip() not in ('', '0')
    try:
        hb = json.load(open(hbp))
        line = f"beats={{hb['beats']}} running={{list(hb['running'])}} done={{hb['done']}} failed={{hb['failed']}}"
    except Exception as e:
        line = f'heartbeat not yet ({{e}})'
    tail = open('/content/worker.log').read().splitlines()[-2:]
    print(time.strftime('%H:%M:%S'), 'worker alive' if alive else 'WORKER STOPPED', '|', line)
    for t in tail:
        print('   ', t[:160])
    if not alive:
        print('ワーカーが止まりました。セル5 で止めた場合は正常です。そうでなければ セル3 から起動し直してください。')
        break
    time.sleep(60)
"""

CELL5 = """#@title 5. 終了（今の仕事が終わってから止まります）
open(f'{{BASE}}/workers/{{WORKER_ID}}/STOP', 'w').write('stop')
print('STOP を置きました。見張りに WORKER STOPPED が出たら、ランタイムを解放してください（ランタイム → ランタイムを接続解除して削除）。')
"""


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


for wid, rt, lanes, acc in [("w01", "CPU（ハイメモリ）", 6, None), ("w02", "CPU（ハイメモリ）", 6, None),
                            ("g01", "GPU（A100 か L4）", 2, "GPU")]:
    cells = [cell(MD.format(wid=wid, rt=rt), "markdown")]
    for src in (CELL1, CELL2, CELL3, CELL4, CELL5):
        s = src.format(wid=wid, lanes=lanes, worker=WORKER_FILE)
        if s.startswith("#@title"):
            compile(s.replace("from google.colab import drive", "drive = None").replace("drive.mount", "print"),
                    wid, "exec")
        cells.append(cell(s))
    meta = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    if acc:
        meta["accelerator"] = acc
    nb = {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}
    path = f"{D}/pai_battery_worker_{wid}.ipynb"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path)
compile(open(f"{D}/tools/{WORKER_FILE}", encoding="utf-8").read(), WORKER_FILE, "exec")
assert b"\r" not in open(f"{D}/tools/{WORKER_FILE}", "rb").read()
print("compile ok")

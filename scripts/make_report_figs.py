"""Build the figures for the report and the data card for the video from the Colab results."""
import glob
import io
import json
import os
import sys
import tarfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.trash_scene import CLASSES  # noqa: E402
from video.cards import data_card, pipeline_card, results_card  # noqa: E402

R = os.environ.get("PAI_COLAB_BASE", os.path.expanduser("~/GoogleDrive/colab_jobs/pai_battery")) + "/results"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "report")
os.makedirs(OUT, exist_ok=True)

# 1) pipeline
pipeline_card().save(f"{OUT}/fig_pipeline.png")

# 2) labelled training images (2x2) + one for the video card
t = tarfile.open(f"{R}/b01_obbgen/out/obb_ds.tar")
imgs, labs = {}, {}
for mem in t:
    if "/train/" not in mem.name:
        continue
    k = os.path.basename(mem.name)[:-4]
    if mem.name.endswith(".jpg") and len(imgs) < 12:
        imgs[k] = t.extractfile(mem).read()
    elif mem.name.endswith(".txt"):
        labs[k] = t.extractfile(mem).read().decode()
    if len(imgs) >= 12 and all(i in labs for i in imgs):
        break
tiles = []
for k in [k for k in imgs if k in labs][:4]:
    im = Image.open(io.BytesIO(imgs[k])).convert("RGB")
    dr = ImageDraw.Draw(im)
    for line in labs[k].splitlines():
        v = line.split()
        c = int(v[0])
        pts = [(float(v[1 + 2 * j]) * 640, float(v[2 + 2 * j]) * 480) for j in range(4)]
        dr.polygon(pts, outline=(255, 30, 30) if c < 2 else (0, 220, 255), width=3 if c < 2 else 2)
    tiles.append(im)
sheet = Image.new("RGB", (1280, 960))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % 2) * 640, (i // 2) * 480))
sheet.save(f"{OUT}/fig_dataset.jpg", quality=90)

# 3) detector predictions on unseen validation images (2x2)
preds = sorted(glob.glob(f"{R}/b03_obbtrain/out/yolo_obb/pred/*.jpg"))
sheet = Image.new("RGB", (1280, 960))
for i, p in enumerate(preds[:4]):
    sheet.paste(Image.open(p).convert("RGB").resize((640, 480)), ((i % 2) * 640, (i // 2) * 480))
sheet.save(f"{OUT}/fig_pred.jpg", quality=90)
data_card(tiles[0], Image.open(preds[1])).save(f"{OUT}/card_data.png")

# 4) results
vis = json.load(open(f"{R}/e02_vis/out/vis.json", encoding="utf-8"))
rows = {}
for job, name in (("e05_e2e_yolo", "AI検出"), ("d08_e2e_oracle", "完璧な検出")):
    rr = [json.loads(l) for p in glob.glob(f"{R}/{job}/out/e2e_*.jsonl") for l in open(p)]
    rows[name] = rr
e2e = {n: (sum(r["hazard_removed"] for r in rr), sum(r["hazard_total"] for r in rr)) for n, rr in rows.items()}
results_card({k: v["recall"] for k, v in vis["recall_by_visibility"].items()}, e2e).save(f"{OUT}/fig_results.png")
summary = {"vis": {k: v for k, v in vis.items() if k != "rows"}, "e2e": {}}
for n, rr in rows.items():
    kinds = {}
    for r in rr:
        for k, s in r["hazards"]:
            kinds.setdefault(k, [0, 0])
            kinds[k][1] += 1
            kinds[k][0] += s == "hazard_box"
    left = {}
    for r in rr:
        for k, s in r["hazards"]:
            if s != "hazard_box":
                key = f"{s}/{r['end']}"
                left[key] = left.get(key, 0) + 1
    summary["e2e"][n] = {"removed": e2e[n][0], "total": e2e[n][1], "trays": len(rr),
                         "all_removed_trays": sum(r["hazard_removed"] == r["hazard_total"] for r in rr),
                         "wrong_box": sum(r["trash_in_box"] for r in rr),
                         "by_kind": kinds, "left_by_status_end": left,
                         "actions_per_tray": sum(r["actions"] for r in rr) / len(rr),
                         "sim_seconds_per_tray": sum(r["sim_seconds"] for r in rr) / len(rr),
                         "clears": sum(r["clears"] for r in rr), "nudges": sum(r["nudges"] for r in rr),
                         "searches": sum(r["searches"] for r in rr)}
json.dump(summary, open(f"{OUT}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 5) demo 2 sequence (uncover then pick)
clip = f"{R}/f02_video/out/scene_1002.mp4"
rd = imageio.get_reader(clip)
n = rd.count_frames()
frames = [Image.fromarray(rd.get_data(int(n * f))) for f in (0.02, 0.30, 0.55, 0.97)]
rd.close()
sheet = Image.new("RGB", (1280, 720))
for i, im in enumerate(frames):
    sheet.paste(im.resize((640, 360)), ((i % 2) * 640, (i // 2) * 360))
sheet.save(f"{OUT}/fig_demo2.jpg", quality=90)
print(json.dumps(summary["e2e"], ensure_ascii=False, indent=1))

"""How well does the detector find batteries that are partly hidden?

For every hazard in many random trays we measure its visible fraction (pixels seen in the
real image / pixels when the other objects are removed) and whether the detector found it
(same class, box centre within 12 px). Also counts false alarms: non-hazards reported as
hazards (pens taken for batteries, candy boxes for power banks ...).

  python perception/eval_visibility.py --weights best.pt --seeds 3000000 3000300 --out out/vis.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perception.detector_source import H, W, YoloObb  # noqa: E402
from sim.trash_scene import CLASSES, HAZARD, PARK, TrashEnv, seg_renderer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--seeds", type=int, nargs=2, default=[3_000_000, 3_000_300])
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--out", required=True)
a = ap.parse_args()

env = TrashEnv(n_hazard=(1, 3), n_trash=(4, 14))
m, d = env.model, env.data
rgb_r = mujoco.Renderer(m, H, W)
seg_r = seg_renderer(m, H, W)
det = YoloObb(a.weights, conf=a.conf)
rows, false_alarms, n_images = [], {}, 0


def seg_ids():
    seg_r.update_scene(d, camera="top_cam")
    s = seg_r.render()
    return np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)


for seed in range(*a.seeds):
    info = env.reset(seed)
    n_images += 1
    rgb_r.update_scene(d, camera="top_cam")
    dets = det(rgb_r.render(), "top_cam")
    ids = seg_ids()
    matched = set()
    for it in info.items:
        mask = np.isin(ids, it.geoms)
        vis_px = int(mask.sum())
        if it.cls not in HAZARD:
            continue
        # unoccluded area: move every other object away, render, put them back
        saved = d.qpos.copy()
        for o in info.items:
            if o is not it:
                qa = env.free_qadr[o.name]
                d.qpos[qa:qa + 3] = PARK + np.array([0, 0.5, 0])
        mujoco.mj_kinematics(m, d)
        full_px = int(np.isin(seg_ids(), it.geoms).sum())
        d.qpos[:] = saved
        mujoco.mj_kinematics(m, d)
        if full_px == 0:
            continue
        ys, xs = np.nonzero(mask) if vis_px else (np.array([0]), np.array([0]))
        cx, cy = (xs.mean(), ys.mean()) if vis_px else (-1e9, -1e9)
        found = False
        for k, (c, p, bx, by, bw, bh, ang) in enumerate(dets):
            if c == it.cls and k not in matched and np.hypot(bx - cx, by - cy) < 12 + 0.5 * min(bw, bh):
                matched.add(k)
                found = True
                break
        rows.append({"seed": seed, "kind": it.kind, "visible": vis_px / full_px, "vis_px": vis_px, "found": found})
    # false alarms: hazard detections that match no hazard
    for k, (c, p, bx, by, bw, bh, ang) in enumerate(dets):
        if c in HAZARD and k not in matched:
            px = (int(np.clip(by, 0, H - 1)), int(np.clip(bx, 0, W - 1)))
            under = [o.kind for o in info.items if ids[px] in o.geoms]
            key = f"{CLASSES[c]}<-{under[0] if under else 'background'}"
            false_alarms[key] = false_alarms.get(key, 0) + 1

bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
summary = {}
for lo, hi in bins:
    sel = [r for r in rows if lo <= r["visible"] < hi]
    summary[f"{int(lo * 100)}-{min(100, int(hi * 100))}%"] = {
        "n": len(sel), "recall": (sum(r["found"] for r in sel) / len(sel)) if sel else None}
out = {"images": n_images, "hazards": len(rows), "recall_by_visibility": summary,
       "false_alarms": false_alarms, "false_alarms_per_image": sum(false_alarms.values()) / max(1, n_images),
       "rows": rows}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

"""End-to-end evaluation of the battery-picking loop over many random trays.

  python scripts/eval_e2e.py --mode yolo:path/best.pt --seeds 1000 1010 --out out/e2e_yolo_0.jsonl
mode: gt (simulator poses) | oracle (perfect boxes from segmentation + depth) | yolo:<weights>
One JSON line per tray: hazards removed / total, actions, non-hazards put in the hazard box, time.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perception.detector_source import DetectorSource, OracleObb  # noqa: E402
from sim.declutter import ClearAndPick, GroundTruthSource  # noqa: E402
from sim.trash_scene import HAZARD, TrashEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="gt")
ap.add_argument("--seeds", type=int, nargs=2, default=[1000, 1010])
ap.add_argument("--trash", type=int, nargs=2, default=[6, 10])
ap.add_argument("--search", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
env = TrashEnv(n_hazard=(1, 2), n_trash=tuple(a.trash))
det = None
if a.mode.startswith("yolo:"):
    from perception.detector_source import YoloObb
    det = YoloObb(a.mode.split(":", 1)[1])
with open(a.out, "w") as f:
    for seed in range(*a.seeds):
        t0 = time.time()
        env.reset(seed)
        if a.mode == "gt":
            src = GroundTruthSource(env)
        elif a.mode == "oracle":
            src = DetectorSource(env, OracleObb(env))
        else:
            src = DetectorSource(env, det)
        cp = ClearAndPick(env, src, search_budget=a.search)
        cp.run()
        env.update_status()
        s = env.summary()
        kinds = [x[0] for x in cp.log]
        rec = dict(seed=seed, mode=a.mode, **s, actions=sum(k in ("pick_hazard", "clear", "search", "nudge") for k in kinds),
                   nudges=kinds.count("nudge"),
                   clears=kinds.count("clear"), searches=kinds.count("search"), end=kinds[-1] if kinds else "",
                   hazards=[(it.kind, it.status) for it in env.info.items if it.cls in HAZARD],
                   sim_seconds=round(env.data.time, 1), wall_seconds=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)

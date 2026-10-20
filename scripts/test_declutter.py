"""Gate test: clear-and-pick with ground-truth object poses over many cluttered trays."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.declutter import ClearAndPick, GroundTruthSource  # noqa: E402
from perception.detector_source import DetectorSource, OracleObb  # noqa: E402
from sim.trash_scene import TrashEnv  # noqa: E402

a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10)
n_trash = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (6, 10)
mode = sys.argv[5] if len(sys.argv) > 5 else "gt"          # gt | oracle | yolo:<weights>
env = TrashEnv(n_hazard=(1, 2), n_trash=n_trash)
tot = {"hazard_total": 0, "hazard_removed": 0, "trash_in_box": 0, "hazard_in_bin": 0, "spilled": 0, "actions": 0}
t0 = time.time()
for seed in range(a, b):
    env.reset(seed)
    if mode == "gt":
        src = GroundTruthSource(env)
    elif mode == "oracle":
        src = DetectorSource(env, OracleObb(env))
    else:
        from perception.detector_source import YoloObb
        src = DetectorSource(env, YoloObb(mode.split(":", 1)[1]))
    cp = ClearAndPick(env, src)
    ok = cp.run()
    env.update_status()
    s = env.summary()
    acts = [x[0] for x in cp.log if x[0] in ("pick_hazard", "clear", "search")]
    for k in s:
        tot[k] += s[k]
    tot["actions"] += len(acts)
    print(f"{seed}: {s['hazard_removed']}/{s['hazard_total']} removed, {acts.count('clear')} cleared, {acts.count('search')} searched, "
          f"end={cp.log[-1][0]} wrong_box={s['trash_in_box']} hz_in_bin={s['hazard_in_bin']} "
          f"spilled={s['spilled']}", flush=True)
print(tot, f"{time.time() - t0:.0f}s")

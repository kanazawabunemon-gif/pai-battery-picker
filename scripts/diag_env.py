"""Print the intermediate values of one look (depth, height map, detected objects, grasp choice).
Run it on two machines and diff the outputs to find where they disagree."""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perception.detector_source import DetectorSource, OracleObb  # noqa: E402
from sim.declutter import ClearAndPick, GroundTruthSource  # noqa: E402
from sim.trash_scene import HAZARD, TrashEnv  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
env = TrashEnv(n_hazard=(1, 2), n_trash=(6, 10))
env.reset(seed)
m, d = env.model, env.data
print("mujoco", mujoco.__version__, "numpy", np.__version__, "MUJOCO_GL", os.environ.get("MUJOCO_GL"))
print("qpos hash", float(np.round(d.qpos.sum(), 6)), "items", [it.name for it in env.info.items])
src = DetectorSource(env, OracleObb(env))
rgb, depth = src.look()
print("rgb mean", rgb.mean().round(3), "depth min/med/max", depth.min(), np.median(depth), depth.max())
cp = ClearAndPick(env, src)
hm = cp.heightmap()
print("heightmap min/med/max", hm.h.min(), np.median(hm.h), hm.h.max())
gt = {o.key: o for o in GroundTruthSource(env).objects()}
for o in src.objects():
    print("det", o.key, o.cls, "c", o.centre, "w %.4f top %.4f" % (o.width, o.top),
          "la", None if o.long_axis is None else o.long_axis)
for k, o in gt.items():
    if o.cls in HAZARD:
        print("gt ", k, o.cls, "c", o.centre, "w %.4f top %.4f" % (o.width, o.top))
        gs, (g, why) = cp._grasp(o, hm)
        print("    best clearance %.4f tip %.4f" % (gs[0].clearance, gs[0].tip_z), "feasible:", g is not None, why)
        ex = cp.ex
        ex._last_jd = gs[0].jaw_dir
        q, e, a = ex._ik(env.qpos_arm(), np.r_[o.centre[:2], gs[0].tip_z], gs[0].jaw_dir, o.width, max_tilt=0.3)
        print("    ik q", q, "err %.5f app %.4f jawerr %.4f" % (e, a, ex.jaw_error()))

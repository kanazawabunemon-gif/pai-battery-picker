"""For trays where the clear-and-pick loop gets stuck, print why each top object is not graspable."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perception.detector_source import DetectorSource, OracleObb  # noqa: E402
from sim.declutter import ClearAndPick  # noqa: E402
from sim.trash_scene import CLASSES, TrashEnv  # noqa: E402

env = TrashEnv(n_hazard=(1, 2), n_trash=(6, 10))
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    env.reset(seed)
    src = DetectorSource(env, OracleObb(env))
    cp = ClearAndPick(env, src)
    cp.run()
    print("seed", seed, [x[0] for x in cp.log], flush=True)
    if cp.log[-1][0] != "stuck":
        continue
    objs = src.objects()
    hm = cp.heightmap()
    for o in sorted(objs, key=lambda o: -o.top)[:8]:
        cov = cp._covered(o, objs, hm)
        gs, (g, why) = cp._grasp(o, hm)
        r = "covered by " + ",".join(CLASSES[c.cls] for c in cov) if cov else (why or "ok")
        print("   %-9s c %s top %.3f w %.3f la %s best clr %+.1fmm -> %s" % (
            CLASSES[o.cls], o.centre.round(3), o.top, o.width, "-" if o.long_axis is None else "y",
            gs[0].clearance * 1000, r), flush=True)

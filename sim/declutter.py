"""Clear-and-pick loop: remove every hazard from the tray, moving other waste out of the way.

Each cycle looks at the tray (depth -> height map, objects from an ObjectSource), then
  1. if some hazard can be grasped (both fingers have room) -> pick it -> red hazard box
  2. otherwise find what blocks the most promising hazard (the object under a finger
     footprint, or lying on top of the hazard) -> pick that -> blue waste bin
until no hazard is left or the action budget runs out.

ObjectSource gives a list of Obj (class, centre, long axis, width, top). GroundTruthSource
reads the simulator; the detector-based source (perception/) has the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .grasp_planner import HeightMap, choose_grasp, drag_direction
from .pick_expert import PickExpert, object_grasp_params, open_for
from .trash_scene import BIN_POS, BOX_POS, HAZARD, depth_renderer


@dataclass
class Obj:
    key: str                  # identifier (slot name for ground truth, detection index otherwise)
    cls: int
    centre: np.ndarray
    long_axis: np.ndarray | None
    width: float
    top: float
    length: float | None = None   # extent along the long axis (for end-to-end grasps)


class GroundTruthSource:
    def __init__(self, env):
        self.env = env

    def objects(self, hm=None):
        env, m, d = self.env, self.env.model, self.env.data
        out = []
        for it in env.info.items:
            if it.status != "tray":
                continue
            c, _, w, top, la = object_grasp_params(m, d, it.body, it.geoms[0])
            g, sz = it.geoms[0], m.geom_size[it.geoms[0]]
            length = 2 * sz[1] if m.geom_type[g] == 5 else (2 * max(sz[:3]) if m.geom_type[g] == 6 else w)
            out.append(Obj(it.name, it.cls, c, la, w, top, float(length)))
        return out


class ClearAndPick:
    def __init__(self, env, source, cam="top_cam", max_actions=16, on_frame=None, search_budget=3):
        self.env, self.source = env, source
        self.ex = PickExpert(env)
        self.cam = cam
        self.max_actions = max_actions
        self.depth_r = depth_renderer(env.model, 480, 640)
        self.on_frame = on_frame          # callback(env, phase_label) after every control step (video)
        self.log = []
        self.fails = []                   # [xy, count]: hazards we tried and missed (retry at most twice)
        self.search_budget = search_budget  # extra clears to look under the pile when no hazard is visible
        self.drags = []                     # [xy, count]: objects we slid
        self.nudge_budget = 6               # slides allowed per tray

    def _fail_count(self, obj):
        for xy, n in self.fails:
            if np.linalg.norm(xy - obj.centre[:2]) < 0.025:
                return n
        return 0

    def _note_result(self, obj):
        """After a hazard pick: if a hazard is still where we grasped, count a miss."""
        still = [o for o in self.source.objects() if o.cls in HAZARD
                 and np.linalg.norm(o.centre[:2] - obj.centre[:2]) < 0.025]
        if not still:
            return
        for f in self.fails:
            if np.linalg.norm(f[0] - obj.centre[:2]) < 0.025:
                f[1] += 1
                return
        self.fails.append([obj.centre[:2].copy(), 1])

    def heightmap(self):
        self.depth_r.update_scene(self.env.data, camera=self.cam)
        return HeightMap.from_depth(self.depth_r.render(), self.env.model, self.env.data, self.cam)

    def _grasp(self, obj, hm):
        gs = choose_grasp(hm, obj.centre, obj.long_axis, obj.width, obj.top, k=10, length=obj.length,
                          gap_of=lambda w: 0.013 + 0.079 * open_for(w))
        return gs, self.ex.first_feasible(self.env.qpos_arm(), obj.centre, gs)

    @staticmethod
    def _covered(obj, objs, hm):
        """Objects lying on top of obj: their centre is within obj's footprint and they are higher."""
        r = 0.5 * max(obj.width, 0.03) + 0.012
        return [o for o in objs if o is not obj and np.linalg.norm(o.centre[:2] - obj.centre[:2]) < r
                and o.top > obj.top + 0.004]

    @staticmethod
    def _count(store, xy):
        for f in store:
            if np.linalg.norm(f[0] - xy[:2]) < 0.03:
                return f
        return None

    def _drag(self, obj, hm, direction=None, label="nudge"):
        if direction is None:
            direction, _ = drag_direction(hm, obj.centre, obj.width, dist=0.06)
        cmds, phases = self.ex.plan_drag(self.env.qpos_arm(), obj.centre, obj.top, direction, 0.06)
        for c, ph in zip(cmds, phases):
            self.env.step(c)
            if self.on_frame:
                self.on_frame(self.env, f"{label}:{ph}")
        f = self._count(self.drags, obj.centre)
        if f:
            f[1] += 1
        self.drags.append([obj.centre[:2] + 0.05 * np.asarray(direction[:2]), 1])

    def _try_remove(self, b, objs, hm, label=None, reason=""):
        """Pick b out of the way (waste bin; hazard box if it is a hazard). False if not possible."""
        if self._covered(b, objs, hm):
            return False
        _, (g, why) = self._grasp(b, hm)
        if g is None:
            return False
        haz = b.cls in HAZARD
        kind = label or ("pick_hazard" if haz else "clear")
        self.log.append((kind, b.key, b.cls, reason))
        self._execute(b, g, (BOX_POS if haz else BIN_POS) + np.array([0, 0, 0.13]),
                      "hazard" if haz else (label or "clear"))
        return True

    def _execute(self, obj, g, place, label):
        cmds, phases, notes = self.ex.plan(self.env.qpos_arm(), obj.centre, g.jaw_dir, g.width,
                                           place=place, tip_z=g.tip_z)
        for c, ph in zip(cmds, phases):
            self.env.step(c)
            if self.on_frame:
                self.on_frame(self.env, f"{label}:{ph}")

    def run(self):
        for step in range(self.max_actions):
            objs = self.source.objects()
            hazards = [o for o in objs if o.cls in HAZARD]
            hm = self.heightmap()
            if not hazards:
                # nothing visible: a battery may be completely buried. Dig a little: remove the
                # tallest object (most likely to hide something) and look again.
                others = sorted((o for o in objs if o.top > 0.03), key=lambda o: -o.top)
                if self.search_budget > 0 and any(self._try_remove(b, objs, hm, label="search") for b in others):
                    self.search_budget -= 1
                    continue
                self.log.append(("done", None, None))
                return True
            # 1) a hazard we can take right now (nothing on it, both fingers have room)
            options = []
            for h in hazards:
                if self._covered(h, objs, hm) or self._fail_count(h) >= 2:
                    continue
                gs, (g, why) = self._grasp(h, hm)
                if g is not None:
                    options.append((g.clearance, h, g))
            if options:
                _, h, g = max(options, key=lambda t: t[0])
                self.log.append(("pick_hazard", h.key, h.cls))
                self._execute(h, g, BOX_POS + np.array([0, 0, 0.13]), "hazard")
                self._note_result(h)
                continue
            # 2) otherwise make room around the most exposed hazard
            acted = False
            for h in sorted(hazards, key=lambda o: -o.top):
                cov = sorted(self._covered(h, objs, hm), key=lambda o: -o.top)
                if cov:                                   # something lies on it: lift that off
                    if any(self._try_remove(b, objs, hm, reason="on_top") for b in cov):
                        acted = True
                        break
                    d = cov[0].centre[:2] - h.centre[:2]
                    if self._count(self.drags, cov[0].centre) is None:
                        self.log.append(("nudge", cov[0].key, cov[0].cls))
                        self._drag(cov[0], hm, d / (np.linalg.norm(d) + 1e-9))   # slide it off
                        acted = True
                        break
                    continue
                gs, _ = self._grasp(h, hm)
                # whatever sits where the best finger would land (any height: packed pens and caps
                # block as much as a can does) is moved to the waste bin first
                bxy = gs[0].block_xy
                near = sorted((o for o in objs if o is not h and np.linalg.norm(o.centre[:2] - bxy)
                               < 0.5 * max(o.width, o.length or o.width) + 0.012),
                              key=lambda o: np.linalg.norm(o.centre[:2] - bxy))
                if any(self._try_remove(b, objs, hm, reason="blocks_fingers") for b in near[:2]):
                    acted = True
                    break
                # nothing removable there: a wall (or an ungraspable object). Slide the hazard out,
                # with a budget for the whole tray so a stubborn case cannot eat every action.
                if self.nudge_budget > 0:
                    self.nudge_budget -= 1
                    self.log.append(("nudge", h.key, h.cls))
                    self._drag(h, hm)
                    acted = True
                    break
            if acted:
                continue
            # 3) last resort: take whatever is on top of the pile
            if any(self._try_remove(b, objs, hm, reason="top_of_pile") for b in sorted(objs, key=lambda o: -o.top)
                   if b.cls not in HAZARD):
                continue
            self.log.append(("stuck", None, None))
            return False
        return not [o for o in self.source.objects() if o.cls in HAZARD]

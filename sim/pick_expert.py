"""Top-down pick-and-place for the SO-101: above -> descend -> pinch -> lift -> carry -> release -> home.

Only needs the object's position, the horizontal direction in which the jaws should
close (across the object's short side) and its width along that direction. These come
from ground truth (demos/tests) or from the detector + depth camera (deployment).
"""
from __future__ import annotations

import numpy as np

from .ik import GripperIK
from .trash_scene import BOX_POS, DT_CONTROL, HOME_Q

DOWN = np.array([0.0, 0.0, -1.0])
FLOOR_Z = 0.010           # tray floor surface
FIXED_PAD_Z = -0.0195     # fixed-jaw pad surface, gripperframe z
G_OPEN, G_CLOSE = 0.75, -0.10


def open_for(width):
    """Gripper angle that opens the jaws ~8 mm wider than the object on each side (clutter-friendly)."""
    return float(np.clip((width + 0.003) / 0.079, 0.15, G_OPEN))


def min_jerk(q0, q1, n):
    s = np.linspace(0, 1, n + 1)[1:]
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0[None] + (q1 - q0)[None] * s[:, None]


def object_grasp_params(model, data, body, geom):
    """Ground-truth grasp parameters of one object: centre, jaw direction, width, top height."""
    R = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    gtype = model.geom_type[geom]
    c = data.geom_xpos[geom].copy()
    if gtype == 5:     # cylinder: long axis = local z
        axis = R[:, 2]
        half = np.array([size[0], size[0], size[1]])
    elif gtype == 6:   # box: long axis = largest half size
        half = size[:3].copy()
        axis = R[:, int(np.argmax(half))]
    else:              # mesh / blob: use the bounding sphere, any direction
        half = np.full(3, model.geom_rbound[geom] * 0.8)
        axis = np.array([1.0, 0, 0])
    h = np.array([axis[0], axis[1], 0.0])
    if np.linalg.norm(h) < 0.3 or gtype not in (5, 6):   # standing upright: no preferred direction
        jd = np.array([-c[1], c[0], 0.0])                 # close tangentially (keeps the wrist simple)
    else:
        jd = np.array([-h[1], h[0], 0.0])
    jd /= np.linalg.norm(jd)
    if gtype == 5:
        a = axis
        width = 2 * (size[0] * np.sqrt(max(0.0, 1 - (a @ jd) ** 2)) + size[1] * abs(a @ jd))
        top = c[2] + size[0] * np.sqrt(max(0.0, 1 - a[2] ** 2)) + size[1] * abs(a[2])
    elif gtype == 6:
        width = 2 * sum(abs(R[:, i] @ jd) * half[i] for i in range(3))
        top = c[2] + sum(abs(R[2, i]) * half[i] for i in range(3))
    else:
        width = 2 * half[0]
        top = c[2] + half[2]
    long_axis = None if (np.linalg.norm(h) < 0.3 or gtype not in (5, 6)) else h / np.linalg.norm(h)
    return c, jd, float(width), float(top), long_axis


class PickExpert:
    def __init__(self, env):
        self.env = env
        self.ik = GripperIK(env.model, env.qadr, env.vadr, np.zeros(3))

    def _ik(self, q_seed, tip, jd, width, max_tilt=0.7, strict=True):
        """Vertical approach if reachable; otherwise tilt the gripper outward (away from the base)
        step by step. High poses close to the base need the tilt (wrist_flex limit)."""
        self.ik.offset = np.array([0.0, 0.0, FIXED_PAD_Z + width / 2 + 0.003])
        out = np.array([tip[0], tip[1], 0.0])
        out /= max(np.linalg.norm(out), 1e-6)
        best = None
        for tilt in np.arange(0.0, max_tilt + 1e-9, 0.15):
            app = np.cos(tilt) * DOWN + np.sin(tilt) * out
            q, e, a = self.ik.solve_robust(q_seed, tip, app, jaw_dir=jd, jaw_strict=strict, allow_flip=True,
                                           restarts=4)
            if best is None or e < best[1]:
                best = (q, e, a)
            if e < 0.0015:
                break
        return best

    def jaw_error(self):
        """Angle [rad] between the jaw-closing axis of the last IK solution and the requested one."""
        ez = self.ik.d.site_xmat[self.ik.site].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(ez @ self._last_jd, -1, 1)))

    def first_feasible(self, q_now, centre, choices, width=None, min_clear=-0.003):
        """First grasp choice (best clearance first) that the arm can actually reach with the
        requested jaw side; returns (choice, None) or (None, reason)."""
        worst = None
        for g in choices:
            if g.clearance < min_clear:
                worst = g if worst is None else worst
                continue
            self._last_jd = g.jaw_dir / np.linalg.norm(g.jaw_dir)
            q, e, _ = self._ik(q_now, np.r_[centre[:2], g.tip_z], g.jaw_dir, g.width if width is None else width,
                               max_tilt=0.3)
            if e < 0.002 and self.jaw_error() < np.deg2rad(8):
                return g, None
        if worst is not None:
            return None, f"blocked ({worst.clearance * 1000:.0f} mm)"
        return None, "no reachable jaw orientation"

    def plan_drag(self, q_now, start, top_z, direction, dist, speed=1.0):
        """Press the closed fingertips onto the object and slide it `dist` along `direction`
        (used when the object sits against a wall or is packed between similar objects)."""
        n = lambda sec: max(2, int(round(sec / (DT_CONTROL * speed))))  # noqa: E731
        cmds, phases = [], []
        q = np.array(q_now, float)
        u = np.array([direction[0], direction[1], 0.0])
        u /= np.linalg.norm(u)
        jd = np.array([-u[1], u[0], 0.0])                 # fingers side by side across the push
        self.ik.offset = np.array([-0.004, 0.0, -0.0195])  # centre of the closed finger pads

        def go(target, sec, grip, phase):
            nonlocal q
            qk, e, _ = self.ik.solve_robust(q, target, DOWN, jaw_dir=jd)
            seg = min_jerk(q[:5], qk, n(sec))
            gs = np.linspace(q[5], grip, len(seg) + 1)[1:]
            for a, gg in zip(seg, gs):
                cmds.append(np.r_[a, gg])
                phases.append(phase)
            q = np.r_[qk, grip]

        p0 = np.array([start[0], start[1], max(FLOOR_Z + 0.006, top_z + 0.001)])   # tips ~3 mm into the top
        go(p0 + np.array([0, 0, 0.06]), 1.4, G_CLOSE, "reach")
        go(p0, 0.6, G_CLOSE, "press")
        for s_ in np.linspace(0, 1, 6)[1:]:
            go(p0 + u * dist * s_, 0.3, G_CLOSE, "drag")
        go(p0 + u * dist + np.array([0, 0, 0.06]), 0.5, G_CLOSE, "lift")
        seg = min_jerk(q[:5], HOME_Q[:5], n(1.3))
        for a in seg:
            cmds.append(np.r_[a, HOME_Q[5]])
            phases.append("home")
        return np.array(cmds), phases

    def plan(self, q_now, centre, jaw_dir, width, place=None, speed=1.0, tip_z=None):
        """Returns (commands [T,6], phases, notes). `place`: drop point (default: hazard box)."""
        n = lambda sec: max(2, int(round(sec / (DT_CONTROL * speed))))  # noqa: E731
        cmds, phases, notes = [], [], {}
        q = np.array(q_now, float)

        def go(q_to, sec, grip, phase):
            nonlocal q
            seg = min_jerk(q[:5], q_to[:5], n(sec))
            gs = np.linspace(q[5], grip, len(seg) + 1)[1:]
            for a, g in zip(seg, gs):
                cmds.append(np.r_[a, g])
                phases.append(phase)
            q = np.r_[q_to[:5], grip]

        def hold(sec, phase, grip=None):
            go(q, sec, q[5] if grip is None else grip, phase)

        width = float(np.clip(width, 0.008, 0.05))
        if tip_z is None:
            tip_z = max(FLOOR_Z + 0.004, centre[2] - 0.006)
        grasp = np.array([centre[0], centre[1], tip_z])
        above = grasp + np.array([0, 0, 0.07])
        g_open = open_for(width)
        q_above, e1, _ = self._ik(q, above, jaw_dir, width)
        go(np.r_[q_above, g_open], 1.6, g_open, "reach")
        q_seed = q_above
        errs = [e1]
        for s in np.linspace(0, 1, 5)[1:]:
            q_k, e, _ = self._ik(q_seed, above + (grasp - above) * s, jaw_dir, width, max_tilt=0.7 * (1 - s) + 0.3)
            errs.append(e)
            go(np.r_[q_k, g_open], 0.25, g_open, "descend")
            q_seed = q_k
        hold(0.6, "grasp", grip=G_CLOSE)
        hold(0.3, "grasp")
        for s in np.linspace(0, 1, 4)[1:]:
            q_k, e, _ = self._ik(q_seed, grasp + np.array([0, 0, 0.09 * s]), jaw_dir, width)
            go(np.r_[q_k, G_CLOSE], 0.25, G_CLOSE, "lift")
            q_seed = q_k
        dst = BOX_POS + np.array([0, 0, 0.13]) if place is None else np.asarray(place, float)
        jd_box = np.array([-dst[1], dst[0], 0.0])
        q_b, eb, _ = self._ik(q_seed, dst, jd_box / np.linalg.norm(jd_box), width, strict=False)
        go(np.r_[q_b, G_CLOSE], 1.4, G_CLOSE, "carry")
        hold(0.5, "release", grip=0.6)
        hold(0.3, "release")
        go(HOME_Q.copy(), 1.4, HOME_Q[5], "home")
        notes["ik_grasp_err"] = float(max(errs))
        notes["ik_place_err"] = float(eb)
        return np.array(cmds), phases, notes

"""Clutter-aware grasp selection from the overhead depth camera.

The depth image is turned into a height map of the tray. For a target (centre, long
axis, width) we try several jaw orientations; for each one we look at where the two
fingers would touch down. If something there is taller than the finger tips, the
fingers would land on it and the grasp fails, so we choose the orientation with the
most clearance. When no orientation is clear, the planner reports which spot blocks
the fingers so the robot can move that object aside first.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .trash_scene import TRAY_HALF, TRAY_POS

FLOOR_Z = 0.010
CELL = 0.002          # height map resolution [m]
JAW_THICK = 0.009     # finger thickness along the closing direction at the tip
JAW_WIDTH = 0.028     # finger width across the closing direction
CLEAR = 0.003         # gap left between pad and object when the gripper descends


class HeightMap:
    def __init__(self):
        self.x0 = TRAY_POS[0] - TRAY_HALF[0] - 0.02
        self.y0 = TRAY_POS[1] - TRAY_HALF[1] - 0.02
        self.nx = int(round((2 * TRAY_HALF[0] + 0.04) / CELL))
        self.ny = int(round((2 * TRAY_HALF[1] + 0.04) / CELL))
        self.h = np.full((self.nx, self.ny), FLOOR_Z)

    @classmethod
    def from_depth(cls, depth, model, data, cam_name):
        hm = cls()
        pts = depth_to_points(depth, model, data, cam_name)
        ix = ((pts[:, 0] - hm.x0) / CELL).astype(int)
        iy = ((pts[:, 1] - hm.y0) / CELL).astype(int)
        ok = (ix >= 0) & (ix < hm.nx) & (iy >= 0) & (iy < hm.ny) & (pts[:, 2] < 0.2)
        h = np.full((hm.nx, hm.ny), -1.0)
        np.maximum.at(h, (ix[ok], iy[ok]), pts[ok, 2])
        # cells that no pixel hit: take the neighbourhood max (small holes at steep edges)
        miss = h < 0
        if miss.any():
            pad = np.pad(h, 1, constant_values=-1)
            stack = np.array([pad[1 + dx:1 + dx + hm.nx, 1 + dy:1 + dy + hm.ny]
                              for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
            stack = np.where(stack < 0, np.inf, stack)
            nb = np.min(stack, axis=0)                 # min, not max: never grow obstacles
            nb = np.where(np.isinf(nb), -1.0, nb)
            h[miss] = nb[miss]
        hm.h = np.where(h < 0, FLOOR_Z, h)
        return hm

    def max_in_rect(self, centre, u, half_u, half_v):
        """Max height (and where) inside the rectangle centred at `centre`, side axis u (unit, xy)."""
        v = np.array([-u[1], u[0]])
        su = np.arange(-half_u, half_u + 1e-9, CELL / 2)
        sv = np.arange(-half_v, half_v + 1e-9, CELL / 2)
        P = centre[None, None, :2] + su[:, None, None] * u[None, None] + sv[None, :, None] * v[None, None]
        P = P.reshape(-1, 2)
        ix = np.clip(((P[:, 0] - self.x0) / CELL).astype(int), 0, self.nx - 1)
        iy = np.clip(((P[:, 1] - self.y0) / CELL).astype(int), 0, self.ny - 1)
        hv = self.h[ix, iy]
        k = int(np.argmax(hv))
        return float(hv[k]), P[k]


def depth_to_points(depth, model, data, cam_name):
    cam = model.camera(cam_name).id
    H, W = depth.shape
    f = 0.5 * H / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = depth
    xc = (u - W / 2) / f * z
    yc = -(v - H / 2) / f * z
    pc = np.stack([xc, yc, -z], -1).reshape(-1, 3)
    R = data.cam_xmat[cam].reshape(3, 3)
    return pc @ R.T + data.cam_xpos[cam]


@dataclass
class GraspChoice:
    jaw_dir: np.ndarray
    tip_z: float
    clearance: float          # >0: both fingers can reach tip_z; <0: blocked by this many metres
    block_xy: np.ndarray      # where the worst finger footprint is highest
    width: float


def choose_grasp(hm: HeightMap, centre, long_axis, width, top_z, open_gap=None, k=1, length=None,
                 gap_of=None, max_span=0.058):
    """Rank jaw orientations by finger clearance. Elongated objects are pinched across their
    short side (also tilted up to 36 deg, which spans width/cos), or end to end if short enough
    (lets the fingers run parallel to a wall). Each choice carries the span the jaws must open to."""
    tip_z = max(FLOOR_Z + 0.008, top_z - 0.02)   # the fixed-jaw rib reaches 4 mm below the tip site
    cand = []                                     # (jaw direction, span)
    if long_axis is None:
        cand = [(np.array([np.cos(a), np.sin(a)]), width) for a in np.deg2rad(np.arange(0, 180, 30))]
    else:
        la = np.asarray(long_axis[:2], float) / np.linalg.norm(long_axis[:2])
        perp = np.array([-la[1], la[0]])
        for deg in (0, 12, -12, 24, -24, 36, -36):
            a = np.deg2rad(deg)
            c, s_ = np.cos(a), np.sin(a)
            span = width / c if length is None else min(width / c, length)
            cand.append((np.array([c * perp[0] - s_ * perp[1], s_ * perp[0] + c * perp[1]]), span))
        if length is not None and length <= max_span:
            cand.append((la, length))
    cands = []
    for jd, span in cand:
        gap = gap_of(span) if gap_of is not None else (open_gap if open_gap is not None else span + 0.016)
        for sign in (1, -1):
            j = sign * jd
            # fixed finger sits CLEAR behind the object on the -j side, moving finger opens to +j
            fc = np.asarray(centre[:2]) - j * (span / 2 + CLEAR + JAW_THICK / 2)
            mc = np.asarray(centre[:2]) - j * (span / 2 + CLEAR) + j * (gap + JAW_THICK / 2)
            hf, pf = hm.max_in_rect(fc, j, JAW_THICK / 2, JAW_WIDTH / 2)
            hmv, pm = hm.max_in_rect(mc, j, JAW_THICK / 2, JAW_WIDTH / 2)
            worst, where = (hf, pf) if hf >= hmv else (hmv, pm)
            clearance = tip_z - worst
            # prefer the plain pinch across the short side a little (most stable)
            score = clearance - 0.001 * float(np.arccos(np.clip(abs(j @ cand[0][0]), -1, 1)))
            cands.append((score, GraspChoice(np.r_[j, 0.0], tip_z, clearance, where, float(span))))
    cands.sort(key=lambda t: -t[0])
    return cands[0][1] if k == 1 else [c for _, c in cands[:k]]


def drag_direction(hm: HeightMap, centre, width, dist=0.05):
    """Direction to slide an object so it ends up with room around it: along each candidate
    direction look at the strip the object would sweep through plus a margin, take its max
    height; prefer low strips and directions towards the tray centre (away from the walls)."""
    best = None
    to_c = TRAY_POS[:2] - np.asarray(centre[:2])
    to_c = to_c / (np.linalg.norm(to_c) + 1e-9)
    for a in np.deg2rad(np.arange(0, 360, 30)):
        u = np.array([np.cos(a), np.sin(a)])
        mid = np.asarray(centre[:2]) + u * (width / 2 + 0.006 + dist / 2 + 0.01)
        h, _ = hm.max_in_rect(mid, u, dist / 2 + 0.01, max(width, 0.02) / 2 + 0.012)
        score = h - 0.01 * float(u @ to_c)
        if best is None or score < best[0]:
            best = (score, u, h)
    return best[1], best[2]

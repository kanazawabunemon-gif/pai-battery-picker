"""Damped-least-squares IK for the SO-101 gripper.

Task: put the grasp point (between the jaws) at a target position, with the approach
axis (gripper x) along a desired direction and the jaw-closing axis (gripper z)
horizontal (side grasps), or a given horizontal direction for top-down grasps.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM = 5   # pan, lift, elbow, wrist_flex, wrist_roll (gripper excluded)


class GripperIK:
    def __init__(self, model: mujoco.MjModel, qadr, vadr, grasp_offset):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.asarray(qadr[:ARM])
        self.vadr = np.asarray(vadr[:ARM])
        self.site = model.site("gripperframe").id
        self.offset = np.asarray(grasp_offset, float)   # grasp point in the gripperframe site frame
        self.lo = model.jnt_range[[model.dof_jntid[v] for v in self.vadr], 0]
        self.hi = model.jnt_range[[model.dof_jntid[v] for v in self.vadr], 1]

    def grasp_point(self, d=None):
        d = d or self.d
        R = d.site_xmat[self.site].reshape(3, 3)
        return d.site_xpos[self.site] + R @ self.offset, R

    def solve(self, q0, target, approach, iters=200, tol=1e-4, w_rot=0.15, jaw_dir=None, jaw_strict=False):
        """Returns (q[5], position error [m], approach error [rad])."""
        m, d = self.m, self.d
        if jaw_dir is not None:
            jaw_dir = np.asarray(jaw_dir, float) / np.linalg.norm(jaw_dir)
        approach = np.asarray(approach, float) / np.linalg.norm(approach)
        q = np.array(q0[:ARM], float)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, R = self.grasp_point(d)
            ex = R[:, 0]
            ez = R[:, 2]
            e_pos = target - p
            e_app = np.cross(ex, approach)            # rotate x onto approach
            if jaw_dir is None:
                e_roll = np.cross(ez, ez - np.array([0, 0, ez[2]]))   # keep z horizontal
            else:                                                     # jaws close along +-jaw_dir
                jd = jaw_dir if (jaw_strict or ez @ jaw_dir >= 0) else -jaw_dir
                e_roll = np.cross(ez, jd)
            err = np.r_[e_pos, w_rot * e_app, 0.5 * w_rot * e_roll]
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_app) < 1e-3 and (
                    jaw_dir is None or np.linalg.norm(e_roll) < 1e-3):
                break
            mujoco.mj_jac(m, d, jp, jr, p, m.site_bodyid[self.site])
            J = np.r_[jp[:, self.vadr], w_rot * jr[:, self.vadr], 0.5 * w_rot * jr[:, self.vadr]]
            lam = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        p, R = self.grasp_point(d)
        ang = float(np.arccos(np.clip(R[:, 0] @ approach, -1, 1)))
        return q, float(np.linalg.norm(target - p)), ang

    def solve_robust(self, q0, target, approach, restarts=12, seed=0, allow_flip=False, jaw_dir=None,
                     jaw_strict=False):
        """Try the given seed first (keeps motions continuous), then random restarts.
        Among good solutions prefer the one closest to q0."""
        rng = np.random.default_rng(seed)
        best = None
        seeds = [np.array(q0[:ARM], float)]
        roll0 = q0[4] if len(q0) > 4 else 1.57
        if jaw_strict:          # the wrist roll decides the jaw side: seed it over its whole range
            for r in np.linspace(self.lo[4] + 0.1, self.hi[4] - 0.1, 7):
                s = np.array(q0[:ARM], float)
                s[4] = r
                seeds.append(s)
        for k in range(restarts):
            s = rng.uniform(self.lo, self.hi)
            flip = allow_flip and k % 2 == 1          # the other roll family turns the wrist camera upside down
            s[4] = (roll0 - np.pi if roll0 > 0 else roll0 + np.pi) if flip else roll0
            seeds.append(s)
        for k, s in enumerate(seeds):
            q, e, a = self.solve(s, target, approach, jaw_dir=jaw_dir, jaw_strict=jaw_strict)
            score = e + 0.01 * a + 0.0005 * np.linalg.norm(q - np.asarray(q0[:ARM]))
            if jaw_dir is not None:
                ez = self.d.site_xmat[self.site].reshape(3, 3)[:, 2]
                jd = np.asarray(jaw_dir, float) / np.linalg.norm(jaw_dir)
                c = ez @ jd if jaw_strict else abs(ez @ jd)
                score += 0.01 * float(np.arccos(np.clip(c, -1, 1)))
            if best is None or score < best[0]:
                best = (score, q, e, a)
            if k == 0 and e < 1e-3 and a < 0.05 and jaw_dir is None:
                break
        return best[1], best[2], best[3]

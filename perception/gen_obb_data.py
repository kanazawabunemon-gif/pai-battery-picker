"""Generate an oriented-box (YOLO-OBB) dataset of the waste tray, labelled automatically.

The overhead camera image of a random pile is rendered together with a segmentation
image, so every visible object gets an exact rotated box (cv2.minAreaRect of its
visible pixels) and its class. Partly hidden batteries keep a box as long as at least
MIN_PX pixels are visible: this is what teaches the detector to spot a battery peeking
out from under other waste.

  python perception/gen_obb_data.py --start 0 --n 500 --out out/obb_ds --split train
"""
import argparse
import os
import sys

import cv2
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.trash_scene import TrashEnv, seg_renderer  # noqa: E402

W, H = 640, 480
MIN_PX = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    img_dir = os.path.join(a.out, "images", a.split)
    lab_dir = os.path.join(a.out, "labels", a.split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    env = TrashEnv(n_hazard=(0, 3), n_trash=(4, 14))
    m, d = env.model, env.data
    cam = m.camera("top_cam").id
    pos0, quat0 = m.cam_pos[cam].copy(), m.cam_quat[cam].copy()
    rgb_r = mujoco.Renderer(m, H, W)
    seg_r = seg_renderer(m, H, W)
    counts = np.zeros(8, int)
    for k in range(a.n):
        seed = 2_000_000 + a.start + k
        rng = np.random.default_rng(seed)
        info = env.reset(seed)
        # calibration error of a real mount: a few mm / degrees
        m.cam_pos[cam] = pos0 + rng.normal(0, 0.004, 3)
        dq, q = np.zeros(4), np.zeros(4)
        mujoco.mju_euler2Quat(dq, rng.normal(0, np.deg2rad(1.0), 3), "xyz")
        mujoco.mju_mulQuat(q, quat0, dq)
        m.cam_quat[cam] = q
        mujoco.mj_forward(m, d)
        rgb_r.update_scene(d, camera="top_cam")
        rgb = rgb_r.render()
        seg_r.update_scene(d, camera="top_cam")
        seg = seg_r.render()
        ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        lines = []
        for it in info.items:
            mask = np.isin(ids, it.geoms)
            n = int(mask.sum())
            if n < MIN_PX:
                continue
            ys, xs = np.nonzero(mask)
            box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)))
            box[:, 0] = np.clip(box[:, 0] / W, 0, 1)
            box[:, 1] = np.clip(box[:, 1] / H, 0, 1)
            lines.append(f"{it.cls} " + " ".join(f"{v:.6f}" for v in box.ravel()))
            counts[it.cls] += 1
        name = f"img_{seed}"
        Image.fromarray(rgb).save(os.path.join(img_dir, name + ".jpg"), quality=92)
        with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if k % 100 == 0:
            print(f"{k}/{a.n} boxes per class so far {counts.tolist()}", flush=True)
    m.cam_pos[cam], m.cam_quat[cam] = pos0, quat0
    print(f"done {a.n} images, boxes per class {counts.tolist()}")


if __name__ == "__main__":
    main()

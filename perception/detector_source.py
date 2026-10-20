"""Object source driven by the overhead camera: oriented boxes + depth -> objects in 3-D.

A detector returns oriented boxes (class, confidence, cx, cy, w, h, angle in pixels).
With the depth image each box becomes an Obj for the clear-and-pick loop:
  centre   - the box centre, lifted onto the object's top surface
  long_axis- the long side of the box, as a horizontal direction in the world
  width    - the short side in metres (what the jaws have to span)
  top      - the highest point inside the box (from depth)
Two detectors share this interface: YoloObb (the trained model) and OracleObb (boxes from
the simulator's segmentation, used to test this conversion without a model).
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from sim.declutter import Obj
from sim.trash_scene import depth_renderer, seg_renderer
from sim.grasp_planner import depth_to_points

W, H = 640, 480


class OracleObb:
    """Oriented boxes from the segmentation image (perfect detector, for testing)."""

    def __init__(self, env, min_px=30):
        self.env = env
        self.seg = seg_renderer(env.model, H, W)
        self.min_px = min_px

    def __call__(self, rgb, cam):
        self.seg.update_scene(self.env.data, camera=cam)
        s = self.seg.render()
        ids = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        out = []
        for it in self.env.info.items:
            if it.status != "tray":
                continue
            mask = np.isin(ids, it.geoms)
            if mask.sum() < self.min_px:
                continue
            ys, xs = np.nonzero(mask)
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            out.append((it.cls, 1.0, cx, cy, w, h, np.deg2rad(ang)))
        return out


class YoloObb:
    def __init__(self, weights, conf=0.35, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, rgb, cam):
        r = self.model.predict(rgb[..., ::-1].copy(), imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        xywhr = r.obb.xywhr.cpu().numpy()
        cls = r.obb.cls.cpu().numpy().astype(int)
        conf = r.obb.conf.cpu().numpy()
        return [(int(c), float(p), *map(float, b)) for c, p, b in zip(cls, conf, xywhr)]


class DetectorSource:
    def __init__(self, env, detector, cam="top_cam"):
        self.env, self.detector, self.cam = env, detector, cam
        self.rgb_r = mujoco.Renderer(env.model, H, W)
        self.dep_r = depth_renderer(env.model, H, W)
        self.last = None            # (rgb, detections) of the latest look, for the video overlay

    def look(self):
        d = self.env.data
        self.rgb_r.update_scene(d, camera=self.cam)
        rgb = self.rgb_r.render()
        self.dep_r.update_scene(d, camera=self.cam)
        depth = self.dep_r.render()
        return rgb, depth

    def objects(self):
        m, d = self.env.model, self.env.data
        rgb, depth = self.look()
        dets = self.detector(rgb, self.cam)
        pts = depth_to_points(depth, m, d, self.cam).reshape(H, W, 3)
        objs = []
        for i, (c, p, cx, cy, w, h, ang) in enumerate(dets):
            box = cv2.boxPoints(((cx, cy), (w, h), np.rad2deg(ang)))
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, box.astype(np.int32), 1)
            # shrink a little so the edges (neighbouring objects) do not leak in
            inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) if mask.sum() > 60 else mask
            P = pts[inner.astype(bool)]
            if len(P) < 5:
                continue
            top = float(np.percentile(P[:, 2], 90))
            u, v = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
            centre = pts[v, u].copy()
            if abs(centre[2] - top) > 0.03:          # the centre pixel hit something else: use the box mean
                centre = P.mean(0)
            centre[2] = top - 0.006
            # long side as a world direction: map two pixels along it through the depth image
            lw, sh = (w, h) if w >= h else (h, w)
            th = ang if w >= h else ang + np.pi / 2
            du, dv = np.cos(th), np.sin(th)
            u2 = int(np.clip(round(cx + 6 * du), 0, W - 1))
            v2 = int(np.clip(round(cy + 6 * dv), 0, H - 1))
            u1 = int(np.clip(round(cx - 6 * du), 0, W - 1))
            v1 = int(np.clip(round(cy - 6 * dv), 0, H - 1))
            a = pts[v2, u2] - pts[v1, u1]
            a[2] = 0
            la = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else None
            dist = float(np.linalg.norm(d.cam_xpos[m.camera(self.cam).id] - centre))
            f = 0.5 * H / np.tan(np.deg2rad(m.cam_fovy[m.camera(self.cam).id]) / 2)
            width = sh * dist / f
            if lw / max(sh, 1e-6) < 1.3:           # round-ish: no preferred jaw direction
                la = None
            objs.append(Obj(f"det{i}", c, centre, la, float(width), top, float(lw * dist / f)))
        self.last = (rgb, dets)
        return objs

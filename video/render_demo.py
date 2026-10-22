"""Render a demo clip: main film camera + "AI view" inset (detections) + caption + counters.

  python video/render_demo.py --seed 1003 --weights best.pt --out out/video/scene_1003.mp4
Frames are written at 20 fps (one per control step, real time). Speed-up is done at editing.
"""
import argparse
import os
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perception.detector_source import DetectorSource, OracleObb  # noqa: E402
from sim.declutter import ClearAndPick  # noqa: E402
from sim.trash_scene import HAZARD, TrashEnv  # noqa: E402

FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothB.ttc"]
JP = {"battery": "電池", "powerbank": "モバイルバッテリー", "can": "缶", "pen": "ペン", "cap": "キャップ",
      "paper": "紙くず", "snackbox": "菓子箱", "carton": "紙パック"}
CLASS_KEYS = list(JP)
CAPTION = {
    "hazard": "危険物を回収中 → 赤い箱へ",
    "clear": "上に乗ったゴミをどかしています",
    "search": "見えない電池がないか、掘って確かめています",
    "nudge": "つかめる位置まで、そっとずらしています",
    "look": "AIがトレイを確認中…",
}


def font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_inset(rgb, dets, w=400):
    """Top-camera image with oriented boxes: hazards thick red with labels, others thin grey."""
    img = rgb.copy()
    for c, p, cx, cy, bw, bh, ang in dets:
        box = cv2.boxPoints(((cx, cy), (bw, bh), np.rad2deg(ang))).astype(np.int32)
        if c in HAZARD:
            cv2.polylines(img, [box], True, (255, 40, 30), 3)
        else:
            cv2.polylines(img, [box], True, (190, 200, 210), 1)
    h = int(img.shape[0] * w / img.shape[1])
    pil = Image.fromarray(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    dr = ImageDraw.Draw(pil)
    f = font(18)
    s = w / rgb.shape[1]
    for c, p, cx, cy, bw, bh, ang in dets:
        if c in HAZARD:
            t = f"{JP[CLASS_KEYS[c]]} {p:.2f}"
            x, y = cx * s + 8, cy * s - 26
            dr.rectangle([x - 3, y - 2, x + f.getlength(t) + 3, y + 22], fill=(200, 20, 20))
            dr.text((x, y), t, font=f, fill=(255, 255, 255))
    return pil


class Recorder:
    def __init__(self, env, source, out, size=(1280, 720), every=1):
        self.env, self.source = env, source
        self.W, self.H = size
        self.r = mujoco.Renderer(env.model, self.H, self.W)
        self.writer = imageio.get_writer(out, fps=20, quality=8, macro_block_size=8)
        self.every, self.k = every, 0
        self.f_cap, self.f_hud, self.f_small = font(34), font(26), font(20)

    def frame(self, env, label):
        self.k += 1
        if self.k % self.every:
            return
        self.r.update_scene(env.data, camera="film_cam")
        pil = Image.fromarray(self.r.render())
        if self.source.last is not None:
            inset = draw_inset(*self.source.last)
            x0, y0 = self.W - inset.width - 16, 16
            pil.paste(inset, (x0, y0))
            dr = ImageDraw.Draw(pil)
            dr.rectangle([x0 - 2, y0 - 2, x0 + inset.width + 1, y0 + inset.height + 1], outline=(255, 255, 255), width=2)
            dr.rectangle([x0, y0 + inset.height + 2, x0 + 210, y0 + inset.height + 30], fill=(20, 20, 20))
            dr.text((x0 + 6, y0 + inset.height + 2), "AIの目（真上カメラ）", font=self.f_small, fill=(255, 255, 255))
        dr = ImageDraw.Draw(pil)
        kind = label.split(":")[0]
        cap = CAPTION.get(kind, "")
        if cap:
            tw = self.f_cap.getlength(cap)
            dr.rectangle([24, self.H - 82, 24 + tw + 28, self.H - 28], fill=(0, 0, 0))
            dr.text((38, self.H - 80), cap, font=self.f_cap, fill=(255, 235, 120))
        hz = [it for it in env.info.items if it.cls in HAZARD]
        done = sum(it.status == "hazard_box" for it in hz)
        cleared = sum(it.status == "waste_bin" for it in env.info.items)
        hud = f"回収した危険物 {done}   どかしたゴミ {cleared}"
        dr.rectangle([24, 20, 40 + self.f_hud.getlength(hud), 62], fill=(0, 0, 0))
        dr.text((32, 22), hud, font=self.f_hud, fill=(255, 255, 255))
        self.writer.append_data(np.asarray(pil))

    def hold(self, sec, label="look"):
        for _ in range(int(sec * 20)):
            self.k = 0
            self.frame(self.env, label)

    def close(self):
        self.writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", default=None, help="YOLO-OBB weights; oracle boxes if omitted")
    ap.add_argument("--trash", type=int, nargs=2, default=[6, 10])
    ap.add_argument("--hazards", nargs="*", default=None, help="force hazard kinds, e.g. battery powerbank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_actions", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    env = TrashEnv(n_hazard=(1, 2), n_trash=tuple(a.trash))
    env.reset(a.seed, hazard_kinds=a.hazards)
    if a.weights:
        from perception.detector_source import YoloObb
        det = YoloObb(a.weights)
    else:
        det = OracleObb(env)
    src = DetectorSource(env, det)
    rec = Recorder(env, src, a.out)
    src.objects()                    # first look, so the inset is filled from the start
    rec.hold(1.5)
    cp = ClearAndPick(env, src, max_actions=a.max_actions, on_frame=rec.frame)
    cp.run()
    src.objects()
    rec.hold(2.0, "done")
    rec.close()
    env.update_status()
    print(a.seed, env.summary(), [x[0] + (f"({x[3]})" if len(x) > 3 and x[3] else "") for x in cp.log])


if __name__ == "__main__":
    main()

"""Assemble the submission video: cards + demo clips (sped up, with a "2倍速" badge).

  python video/assemble.py --clips a.mp4:デモ1の見出し b.mp4:デモ2の見出し --vis vis.json \
      --e2e "AI検出=57/76" "完璧な検出=54/76" --account NAME --out final.mp4
"""
import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from video.cards import (pipeline_card, problem_card, results_card, summary_card,  # noqa: E402
                         title_card, to_frames)
from video.render_demo import font  # noqa: E402

FPS = 20


def badge(frame, speed, heading):
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    if speed > 1:
        t = f"{speed:g}倍速"
        f = font(30)
        x, y = im.width - f.getlength(t) - 40, im.height - 70
        dr.rounded_rectangle([x - 14, y - 6, x + f.getlength(t) + 14, y + 44], 10, fill=(200, 30, 30))
        dr.text((x, y), t, font=f, fill=(255, 255, 255))
    if heading:
        f = font(28)
        dr.rectangle([24, 72, 40 + f.getlength(heading), 116], fill=(255, 205, 60))
        dr.text((32, 74), heading, font=f, fill=(20, 20, 20))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="path:heading")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--e2e", nargs="+", required=True, help='"name=removed/total"')
    ap.add_argument("--account", default="")
    ap.add_argument("--data_card", default=None, help="PNG of the training-data card")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vis = json.load(open(a.vis, encoding="utf-8"))["recall_by_visibility"]
    vis = {k: v["recall"] for k, v in vis.items()}
    e2e = {}
    for s in a.e2e:
        name, frac = s.split("=")
        r, t = frac.split("/")
        e2e[name] = (int(r), int(t))
    w = imageio.get_writer(a.out, fps=FPS, quality=8, macro_block_size=8)
    n = 0

    def put(frames):
        nonlocal n
        for f in frames:
            w.append_data(f)
            n += 1

    put(to_frames(title_card(a.account), 5))
    put(to_frames(problem_card(), 9))
    put(to_frames(pipeline_card(), 10))
    if a.data_card:
        put(to_frames(Image.open(a.data_card).convert("RGB"), 8))
    step = int(round(a.speed))
    for spec in a.clips:
        path, heading = spec.split(":", 1)
        r = imageio.get_reader(path)
        for i, fr in enumerate(r):
            if i % step == 0:
                put([badge(fr, a.speed, heading)])
        r.close()
    put(to_frames(results_card(vis, e2e), 11))
    put(to_frames(summary_card(), 7))
    w.close()
    print(f"wrote {a.out}: {n} frames = {n / FPS:.1f} s")


if __name__ == "__main__":
    main()

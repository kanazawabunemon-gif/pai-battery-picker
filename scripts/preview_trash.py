"""Render the battery-picking scene from the overhead and film cameras (development aid)."""
import os
import sys
import time

import mujoco
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.trash_scene import TrashEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "preview_trash")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
env = TrashEnv()
print("model loaded", round(time.time() - t0, 1), "s")
r = mujoco.Renderer(env.model, 360, 480)
tiles = []
for seed in [int(s) for s in sys.argv[1:]] or [0, 1]:
    t0 = time.time()
    info = env.reset(seed)
    print(seed, "settle", round(time.time() - t0, 1), "s items", [(it.name, it.status) for it in info.items])
    for cam in ("top_cam", "film_cam"):
        r.update_scene(env.data, camera=cam)
        tiles.append(r.render())
sheet = Image.new("RGB", (480 * 2, 360 * (len(tiles) // 2)))
for k, im in enumerate(tiles):
    sheet.paste(Image.fromarray(im), ((k % 2) * 480, (k // 2) * 360))
sheet.save(os.path.join(OUT, "sheet.png"))

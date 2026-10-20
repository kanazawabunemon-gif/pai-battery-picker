#!/bin/bash
# Make EGL use the NVIDIA driver on Colab (otherwise MuJoCo silently falls back to Mesa software rendering).
# Same fix as MuJoCo's official Colab tutorial. Must run before the first renderer is created.
F=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ ! -f "$F" ] && nvidia-smi > /dev/null 2>&1; then
  mkdir -p /usr/share/glvnd/egl_vendor.d
  printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : { "library_path" : "libEGL_nvidia.so.0" }\n}\n' > "$F"
  echo "wrote $F"
fi
python - <<'PY'
import time, mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 480, 640); d = mujoco.MjData(m)
t = time.time()
for _ in range(20):
    r.update_scene(d); r.render()
print("egl check: ms/frame", round((time.time() - t) / 20 * 1000, 2))
PY

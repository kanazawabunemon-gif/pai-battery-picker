"""Battery-picking scene: a sorting tray of household waste with batteries hidden in it.

Lithium batteries thrown out with ordinary waste start fires in garbage trucks and
sorting plants. The SO-101 looks at the tray from an overhead camera, finds the
batteries / power banks, uncovers them if other waste lies on top, and drops them
into the red hazard box.

The model holds a fixed set of object "slots" per type. reset() decides which slots
are used, drops them into the tray from random poses and lets physics settle the pile.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO101_DIR = os.path.join(HERE, "assets", "so101")
DT_CONTROL = 0.05
GRIP_TORQUE = 0.8
PARK = np.array([2.0, 2.0, 0.05])

TRAY_POS = np.array([0.25, 0.0, 0.0])
TRAY_HALF = np.array([0.10, 0.13, 0.03])      # inner half sizes x, y and wall half height
BOX_POS = np.array([0.04, -0.235, 0.0])        # hazard box
BOX_HALF = np.array([0.055, 0.055, 0.035])
BIN_POS = np.array([0.04, 0.235, 0.0])         # general-waste bin for objects moved out of the way
HOME_Q = np.array([-1.15, -1.55, 1.45, 0.75, -1.52, 0.0])   # arm parked beside the hazard box

# detection classes (index = YOLO class id); the first two are hazards
CLASSES = ["battery", "powerbank", "can", "pen", "cap", "paper", "snackbox", "carton"]
HAZARD = {0, 1}
# slots per type
COUNTS = {"battery": 3, "powerbank": 2, "can": 3, "pen": 3, "cap": 4, "paper": 4, "snackbox": 2, "carton": 2}

BATTERY_SKINS = [((0.10, 0.10, 0.10), (0.80, 0.55, 0.15)), ((0.15, 0.45, 0.20), (0.85, 0.80, 0.25)),
                 ((0.10, 0.25, 0.65), (0.85, 0.85, 0.85)), ((0.75, 0.10, 0.10), (0.80, 0.80, 0.80)),
                 ((0.85, 0.85, 0.80), (0.20, 0.20, 0.20))]
PB_COLORS = [(0.92, 0.92, 0.92), (0.12, 0.12, 0.14), (0.15, 0.20, 0.40), (0.85, 0.75, 0.80)]
CAN_COLORS = [(0.75, 0.75, 0.78), (0.70, 0.12, 0.12), (0.12, 0.30, 0.65), (0.20, 0.55, 0.30), (0.90, 0.75, 0.20)]
PEN_COLORS = [(0.10, 0.10, 0.10), (0.15, 0.25, 0.70), (0.80, 0.15, 0.15), (0.95, 0.95, 0.95), (0.20, 0.60, 0.25)]
CAP_COLORS = [(0.95, 0.95, 0.95), (0.95, 0.55, 0.10), (0.15, 0.35, 0.80), (0.20, 0.60, 0.25), (0.90, 0.85, 0.20)]
PAPER_COLORS = [(0.93, 0.92, 0.88), (0.80, 0.78, 0.72), (0.88, 0.85, 0.70), (0.70, 0.72, 0.75)]
SNACK_COLORS = [(0.85, 0.20, 0.25), (0.30, 0.20, 0.10), (0.95, 0.80, 0.25), (0.25, 0.55, 0.80)]
CARTON_COLORS = [(0.95, 0.65, 0.15), (0.35, 0.65, 0.25), (0.95, 0.95, 0.95), (0.55, 0.30, 0.55)]


def _paper_mesh(k: int) -> str:
    rng = np.random.default_rng(100 + k)
    n = 40
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.016, 0.026, (n, 1))
    v[:, 2] *= 0.75
    return f'<mesh name="paper_mesh{k}" vertex="{" ".join(f"{x:.4f}" for x in v.ravel())}"/>'


def _obj_xml(kind: str, i: int) -> str:
    name = f"{kind}{i}"
    p = f'{PARK[0] + 0.1 * len(name) + 0.01 * i} {PARK[1]} {PARK[2]}'
    head = f'<body name="{name}" pos="{p}"><freejoint name="{name}_free"/>'
    if kind == "battery":     # AA cell lying on its side: body, label band, + terminal
        g = (f'<geom name="{name}" type="cylinder" size="0.0072 0.0245" mass="0.023" rgba="0.1 0.1 0.1 1"/>'
             f'<geom name="{name}_band" type="cylinder" size="0.00725 0.009" pos="0 0 0.012" mass="0" '
             f'rgba="0.8 0.55 0.15 1" contype="0" conaffinity="0"/>'
             f'<geom name="{name}_nub" type="cylinder" size="0.0026 0.0012" pos="0 0 0.0255" mass="0" '
             f'rgba="0.8 0.8 0.82 1" contype="0" conaffinity="0" material="metal"/>')
    elif kind == "powerbank":
        g = (f'<geom name="{name}" type="box" size="0.034 0.0175 0.0075" mass="0.10" rgba="0.92 0.92 0.92 1"/>'
             f'<geom name="{name}_usb" type="box" size="0.0006 0.0045 0.0018" pos="0.0342 0 0" mass="0" '
             f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>'
             f'<geom name="{name}_led" type="box" size="0.005 0.0012 0.0003" pos="0.015 0 0.0076" mass="0" '
             f'rgba="0.2 0.8 1.0 1" contype="0" conaffinity="0" material="glow"/>')
    elif kind == "can":       # empty, slightly crushed drink can
        g = (f'<geom name="{name}" type="cylinder" size="0.024 0.036" mass="0.015" rgba="0.75 0.75 0.78 1" '
             f'material="metal"/>'
             f'<geom name="{name}_top" type="cylinder" size="0.022 0.0015" pos="0 0 0.0365" mass="0" '
             f'rgba="0.8 0.8 0.82 1" contype="0" conaffinity="0" material="metal"/>')
    elif kind == "pen":       # marker pen: the same diameter as an AA cell, longer, with a cap
        g = (f'<geom name="{name}" type="cylinder" size="0.0065 0.055" mass="0.012" rgba="0.95 0.95 0.95 1"/>'
             f'<geom name="{name}_cap" type="cylinder" size="0.0072 0.016" pos="0 0 0.042" mass="0" '
             f'rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>')
    elif kind == "cap":
        g = f'<geom name="{name}" type="cylinder" size="0.0145 0.0065" mass="0.003" rgba="0.95 0.95 0.95 1"/>'
    elif kind == "paper":
        g = (f'<geom name="{name}" type="mesh" mesh="paper_mesh{i % 4}" mass="0.006" rgba="0.93 0.92 0.88 1" '
             f'material="paper"/>')
    elif kind == "snackbox":  # candy/gum box: the same shape as a power bank
        g = (f'<geom name="{name}" type="box" size="0.031 0.018 0.008" mass="0.02" rgba="0.85 0.2 0.25 1"/>'
             f'<geom name="{name}_label" type="box" size="0.018 0.0182 0.0002" pos="0 0 0.0081" mass="0" '
             f'rgba="0.98 0.95 0.85 1" contype="0" conaffinity="0"/>')
    elif kind == "carton":    # small juice carton lying down
        g = (f'<geom name="{name}" type="box" size="0.036 0.019 0.013" mass="0.012" rgba="0.95 0.65 0.15 1"/>'
             f'<geom name="{name}_label" type="box" size="0.02 0.0192 0.0132" pos="-0.004 0 0" mass="0" '
             f'rgba="0.98 0.98 0.98 1" contype="0" conaffinity="0"/>')
    else:
        raise ValueError(kind)
    return head + g + "</body>"


def build_xml() -> str:
    objs = "\n    ".join(_obj_xml(k, i) for k, n in COUNTS.items() for i in range(n))
    meshes = "\n    ".join(_paper_mesh(k) for k in range(4))
    tx, ty, _ = TRAY_POS
    hx, hy, hz = TRAY_HALF
    bx, by, _ = BOX_POS
    bhx, bhy, bhz = BOX_HALF
    t = 0.005
    return f"""<mujoco model="battery_picking">
  <include file="so101_pads.xml"/>
  <option timestep="0.005"/>
  <default><default class="trash"><geom friction="0.8 0.01 0.001" condim="4" solref="0.005 1"/></default></default>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="135" elevation="-30" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.82 0.84 0.86" rgb2="0.55 0.57 0.6" width="512" height="3072"/>
    <texture type="2d" name="concrete" builtin="flat" mark="random" random="0.15" rgb1="0.62 0.62 0.6"
             markrgb="0.5 0.5 0.48" width="512" height="512"/>
    <material name="concrete" texture="concrete" texrepeat="8 8" reflectance="0.03"/>
    <texture type="2d" name="belt" builtin="checker" rgb1="0.22 0.23 0.25" rgb2="0.25 0.26 0.28" width="64" height="64"/>
    <material name="belt" texture="belt" texrepeat="12 12" reflectance="0.02"/>
    <material name="metal" specular="0.9" shininess="0.9" reflectance="0.1"/>
    <material name="paper" specular="0.05" shininess="0.05"/>
    <material name="glow" emission="0.8"/>
    <material name="tray_wall" rgba="0.35 0.38 0.42 1" specular="0.3"/>
    <material name="hazard" rgba="0.80 0.12 0.10 1" specular="0.3"/>
    <material name="stripe" rgba="0.98 0.82 0.10 1"/>
    <material name="bin" rgba="0.20 0.42 0.70 1" specular="0.3"/>
    {meshes}
  </asset>
  <worldbody>
    <light name="key" pos="0.3 -0.3 1.6" dir="-0.1 0.15 -1" directional="true" castshadow="true" diffuse="0.55 0.55 0.55"/>
    <light name="fill" pos="-0.5 0.6 1.0" dir="0.5 -0.5 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="concrete"/>
    <camera name="top_cam" pos="{tx} 0 0.55" xyaxes="0 -1 0 1 0 0" fovy="27"/>
    <camera name="film_cam" pos="0.78 -0.16 0.46" xyaxes="0.2930 0.9561 0 -0.5198 0.1593 0.8393" fovy="35"/>
    <body name="tray" pos="{tx} {ty} 0">
      <geom name="tray_floor" type="box" size="{hx + t} {hy + t} {t}" pos="0 0 {t}" material="belt"/>
      <geom name="tray_w1" type="box" size="{hx + 2 * t} {t} {hz}" pos="0 {hy + t} {hz}" material="tray_wall"/>
      <geom name="tray_w2" type="box" size="{hx + 2 * t} {t} {hz}" pos="0 {-hy - t} {hz}" material="tray_wall"/>
      <geom name="tray_w3" type="box" size="{t} {hy} {hz}" pos="{hx + t} 0 {hz}" material="tray_wall"/>
      <geom name="tray_w4" type="box" size="{t} {hy} {hz}" pos="{-hx - t} 0 {hz}" material="tray_wall"/>
      <geom name="mast" type="cylinder" size="0.008 0.29" pos="{hx + 0.03} {hy + 0.03} 0.29" rgba="0.4 0.4 0.42 1"
            contype="0" conaffinity="0"/>
      <geom name="mast_arm" type="capsule" fromto="{hx + 0.03} {hy + 0.03} 0.58 0 0 0.58" size="0.006"
            rgba="0.4 0.4 0.42 1" contype="0" conaffinity="0"/>
      <geom name="cam_body" type="box" size="0.02 0.03 0.012" pos="0 0 0.57" rgba="0.15 0.15 0.15 1"
            contype="0" conaffinity="0"/>
    </body>
    <body name="hazard_box" pos="{bx} {by} 0">
      <geom name="hb_floor" type="box" size="{bhx + t} {bhy + t} {t}" pos="0 0 {t}" material="hazard"/>
      <geom name="hb_w1" type="box" size="{bhx + 2 * t} {t} {bhz}" pos="0 {bhy + t} {bhz}" material="hazard"/>
      <geom name="hb_w2" type="box" size="{bhx + 2 * t} {t} {bhz}" pos="0 {-bhy - t} {bhz}" material="hazard"/>
      <geom name="hb_w3" type="box" size="{t} {bhy} {bhz}" pos="{bhx + t} 0 {bhz}" material="hazard"/>
      <geom name="hb_w4" type="box" size="{t} {bhy} {bhz}" pos="{-bhx - t} 0 {bhz}" material="hazard"/>
      <geom name="hb_stripe" type="box" size="{t + 0.0005} {bhy * 0.8} 0.008" pos="{bhx + t} 0 {bhz * 1.3}"
            material="stripe" contype="0" conaffinity="0"/>
    </body>
    <body name="waste_bin" pos="{BIN_POS[0]} {BIN_POS[1]} 0">
      <geom name="wb_floor" type="box" size="{bhx + t} {bhy + t} {t}" pos="0 0 {t}" material="bin"/>
      <geom name="wb_w1" type="box" size="{bhx + 2 * t} {t} {bhz}" pos="0 {bhy + t} {bhz}" material="bin"/>
      <geom name="wb_w2" type="box" size="{bhx + 2 * t} {t} {bhz}" pos="0 {-bhy - t} {bhz}" material="bin"/>
      <geom name="wb_w3" type="box" size="{t} {bhy} {bhz}" pos="{bhx + t} 0 {bhz}" material="bin"/>
      <geom name="wb_w4" type="box" size="{t} {bhy} {bhz}" pos="{-bhx - t} 0 {bhz}" material="bin"/>
    </body>
    <frame childclass="trash">
    {objs}
    </frame>
  </worldbody>
</mujoco>
"""


def _no_msaa_renderer(model, h, w):
    old = model.vis.quality.offsamples
    model.vis.quality.offsamples = 0
    try:
        return mujoco.Renderer(model, h, w)
    finally:
        model.vis.quality.offsamples = old


def seg_renderer(model, h, w):
    """Segmentation renderer WITHOUT multisampling: with MSAA (NVIDIA EGL) the id colours blend
    at object edges and decode to wrong objects, which inflates masks and boxes."""
    r = _no_msaa_renderer(model, h, w)
    r.enable_segmentation_rendering()
    return r


def depth_renderer(model, h, w):
    """Depth renderer without multisampling: resolved MSAA depth mixes near and far surfaces at
    edges, which shows up as phantom heights between objects in the height map."""
    r = _no_msaa_renderer(model, h, w)
    r.enable_depth_rendering()
    return r


def load_model() -> mujoco.MjModel:
    xml = build_xml()
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="_scene_", dir=SO101_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(xml)
        return mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)


@dataclass
class Item:
    kind: str
    cls: int
    name: str
    body: int
    geoms: list
    status: str = "tray"      # tray / hazard_box / out


@dataclass
class EpisodeInfo:
    seed: int
    items: list = field(default_factory=list)


class TrashEnv:
    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, n_hazard=(1, 3), n_trash=(8, 14), randomize_visuals=True):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(DT_CONTROL / m.opt.timestep))
        self.n_hazard, self.n_trash = n_hazard, n_trash
        self.randomize_visuals = randomize_visuals
        self.qadr = np.array([m.joint(j).qposadr[0] for j in self.ARM_JOINTS])
        self.vadr = np.array([m.joint(j).dofadr[0] for j in self.ARM_JOINTS])
        self.gripper_body = m.body("gripper").id
        self.moving_jaw_body = m.body("moving_jaw_so101_v1").id
        for g in range(m.ngeom):      # jaw hulls fill the inside of the jaws: grasp with boxes + pads only
            if (m.geom_bodyid[g] in (self.gripper_body, self.moving_jaw_body)
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self.jaw_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (self.gripper_body, self.moving_jaw_body)
                          and m.geom_contype[g] != 0}
        m.actuator_forcerange[m.actuator("gripper").id] = [-GRIP_TORQUE, GRIP_TORQUE]
        self.slots = []
        for kind, n in COUNTS.items():
            for i in range(n):
                b = m.body(f"{kind}{i}").id
                geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
                self.slots.append(Item(kind, CLASSES.index(kind), f"{kind}{i}", b, geoms))
        self.free_qadr = {it.name: m.joint(f"{it.name}_free").qposadr[0] for it in self.slots}
        self.free_vadr = {it.name: m.joint(f"{it.name}_free").dofadr[0] for it in self.slots}
        self.light_ids = [m.light("key").id, m.light("fill").id]
        self.light_diffuse0 = m.light_diffuse[self.light_ids].copy()
        self.info: EpisodeInfo | None = None

    # ------------------------------------------------------------------ reset
    def _skin(self, it: Item, rng):
        m = self.model
        g0 = it.geoms[0]
        pick = lambda cs: np.r_[np.clip(np.array(cs[int(rng.integers(len(cs)))]) + rng.normal(0, 0.03, 3), 0, 1), 1]  # noqa
        if it.kind == "battery":
            body, band = BATTERY_SKINS[int(rng.integers(len(BATTERY_SKINS)))]
            m.geom_rgba[g0] = np.r_[body, 1]
            m.geom_rgba[it.geoms[1]] = np.r_[band, 1]
            aaa = rng.random() < 0.35                       # AAA cells too
            m.geom_size[g0, :2] = [0.0052, 0.0222] if aaa else [0.0072, 0.0245]
            m.geom_size[it.geoms[1], :2] = [m.geom_size[g0, 0] + 0.00005, 0.009]
            m.geom_pos[it.geoms[2], 2] = m.geom_size[g0, 1] + 0.001
        elif it.kind == "powerbank":
            m.geom_rgba[g0] = pick(PB_COLORS)
        elif it.kind == "can":
            m.geom_rgba[g0] = pick(CAN_COLORS)
        elif it.kind == "pen":
            m.geom_rgba[g0] = pick(PEN_COLORS)
            m.geom_rgba[it.geoms[1]] = pick(PEN_COLORS)
        elif it.kind == "cap":
            m.geom_rgba[g0] = pick(CAP_COLORS)
        elif it.kind == "paper":
            m.geom_rgba[g0] = pick(PAPER_COLORS)
        elif it.kind == "snackbox":
            m.geom_rgba[g0] = pick(SNACK_COLORS)
        elif it.kind == "carton":
            m.geom_rgba[g0] = pick(CARTON_COLORS)

    def reset(self, seed: int, settle_sec: float = 1.5, hazard_kinds=None) -> EpisodeInfo:
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = HOME_Q
        d.ctrl[:6] = HOME_Q
        hz = [it for it in self.slots if it.cls in HAZARD]
        tr = [it for it in self.slots if it.cls not in HAZARD]
        n_h = int(rng.integers(self.n_hazard[0], self.n_hazard[1] + 1))
        n_t = int(rng.integers(self.n_trash[0], self.n_trash[1] + 1))
        if hazard_kinds is not None:
            chosen_h = []
            for k in hazard_kinds:
                chosen_h.append(next(it for it in hz if it.kind == k and it not in chosen_h))
        else:
            chosen_h = list(rng.choice(hz, size=min(n_h, len(hz)), replace=False))
        chosen_t = list(rng.choice(tr, size=min(n_t, len(tr)), replace=False))
        used = chosen_h + chosen_t
        order = list(rng.permutation(len(used)))
        # hazards tend to be dropped first (under the rest) half of the time: hidden batteries
        if rng.random() < 0.5:
            order = sorted(order, key=lambda k: (used[k].cls not in HAZARD, rng.random()))
        for it in self.slots:
            qa = self.free_qadr[it.name]
            d.qpos[qa:qa + 3] = PARK + np.array([0.08 * self.slots.index(it), 0, 0])
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            it.status = "parked"
        hx, hy, _ = TRAY_HALF
        for rank, k in enumerate(order):
            it = used[k]
            self._skin(it, rng)
            qa = self.free_qadr[it.name]
            x = TRAY_POS[0] + rng.uniform(-hx + 0.03, hx - 0.03)
            y = TRAY_POS[1] + rng.uniform(-hy + 0.03, hy - 0.03)
            d.qpos[qa:qa + 3] = [x, y, 0.03 + 0.035 * rank]
            q = rng.normal(size=4)
            d.qpos[qa + 3:qa + 7] = q / np.linalg.norm(q)
            it.status = "tray"
        if self.randomize_visuals:
            m.light_diffuse[self.light_ids] = self.light_diffuse0 * rng.uniform(0.75, 1.25)
        # let the pile settle (arm held at home)
        for _ in range(int(settle_sec / m.opt.timestep)):
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        for it in used:                       # anything that bounced out of the tray is removed
            if not self.in_tray(d.xpos[it.body]):
                qa = self.free_qadr[it.name]
                d.qpos[qa:qa + 3] = PARK + np.array([0.08 * self.slots.index(it), 0.3, 0])
                it.status = "out"
        mujoco.mj_forward(m, d)
        self.info = EpisodeInfo(seed=seed, items=[it for it in used if it.status == "tray"])
        return self.info

    # ------------------------------------------------------------------- step
    def step(self, ctrl):
        m, d = self.model, self.data
        d.ctrl[:6] = ctrl
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
        self.update_status()

    def update_status(self):
        for it in self.info.items:
            p = self.data.xpos[it.body]
            if self.in_box(p):
                it.status = "hazard_box"
            elif self.in_bin(p):
                it.status = "waste_bin"
            elif self.in_tray(p):
                it.status = "tray"
            else:
                it.status = "out"

    @staticmethod
    def in_tray(p) -> bool:
        r = np.abs(np.asarray(p[:2]) - TRAY_POS[:2])
        return bool(r[0] < TRAY_HALF[0] + 0.005 and r[1] < TRAY_HALF[1] + 0.005 and p[2] < 0.15)

    @staticmethod
    def in_bin(p) -> bool:
        r = np.abs(np.asarray(p[:2]) - BIN_POS[:2])
        return bool(r[0] < BOX_HALF[0] and r[1] < BOX_HALF[1] and p[2] < 2 * BOX_HALF[2] + 0.01)

    @staticmethod
    def in_box(p) -> bool:
        r = np.abs(np.asarray(p[:2]) - BOX_POS[:2])
        return bool(r[0] < BOX_HALF[0] and r[1] < BOX_HALF[1] and p[2] < 2 * BOX_HALF[2] + 0.01)

    def qpos_arm(self):
        return self.data.qpos[self.qadr].copy()

    def summary(self) -> dict:
        hz = [it for it in self.info.items if it.cls in HAZARD]
        return {"hazard_total": len(hz), "hazard_removed": sum(it.status == "hazard_box" for it in hz),
                "trash_in_box": sum(it.status == "hazard_box" for it in self.info.items if it.cls not in HAZARD),
                "hazard_in_bin": sum(it.status == "waste_bin" for it in hz),
                "spilled": sum(it.status == "out" for it in self.info.items)}

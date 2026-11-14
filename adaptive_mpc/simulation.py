"""Headless MuJoCo execution and a procedural, mesh-free scene."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .task import PEG, WALLS, TARGET, FINGER_LOWER, FINGER_UPPER, Settings, initial_state


def model_xml(settings=None):
    s = settings or Settings()
    root = ET.Element("mujoco", model="adaptive_geometry_peg")

    def node(parent, tag, **attrs):
        return ET.SubElement(parent, tag, {k: " ".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray))
                                          else str(v) for k, v in attrs.items()})

    node(root, "compiler", angle="radian")
    node(root, "option", timestep=s.physics_step, gravity="0 0 -9.8")
    node(root, "size", njmax=600, nconmax=160)
    world = node(root, "worldbody")
    node(world, "light", pos="0 0 3", dir="0 0 -1", directional="true",
         ambient="0.55 0.55 0.55", diffuse="0.8 0.8 0.8", castshadow="false")
    node(world, "camera", name="overview", pos="0.03 -0.42 0.44", xyaxes="1 0 0 0 0.72 0.69")
    goal = node(world, "body", name="goal", pos=TARGET[:3])
    for j, (center, half) in enumerate(PEG):
        node(goal, "geom", name=f"goal_{j}", type="box", pos=center, size=half,
             rgba="0.1 1 0.1 0.22", contype=0, conaffinity=0)
    peg = node(world, "body", name="peg")
    node(peg, "freejoint", name="peg_pose")
    for j, (center, half) in enumerate(PEG):
        node(peg, "geom", name=f"peg_{j}", type="box", pos=center, size=half,
             mass=(.008, .002)[j], condim=6, friction=.7,
             rgba=("0.9 0.65 0.12 1", "0.95 0.35 0.08 1")[j])
    node(world, "geom", name="table", type="plane", size="0 0 0.05", condim=6,
         friction=.5, rgba="0.18 0.23 0.28 1")
    for j, (center, half, _) in enumerate(WALLS):
        node(world, "geom", name=f"wall_{j}", type="box", pos=center, size=half,
             condim=6, friction=.6, rgba="0.25 0.45 0.75 1")
    actuator = node(root, "actuator")
    for finger, color in enumerate(("0.8 0.1 0.1 1", "0.1 0.8 0.1 1", "0.1 0.25 0.9 1")):
        body = node(world, "body", name=f"finger_{finger}")
        for axis in range(3):
            name = f"finger_{finger}_{axis}"
            node(body, "joint", name=name, type="slide", axis=np.eye(3)[axis])
            node(actuator, "motor", joint=name, ctrlrange="-2 2", ctrllimited="false")
        node(body, "geom", name=f"finger_{finger}", type="sphere", size=.010,
             condim=6, friction=.6, rgba=color)
    return ET.tostring(root, encoding="unicode")


class Simulation:
    def __init__(self, seed=0, settings=None):
        if mujoco.__version__ != "3.1.6":
            raise RuntimeError("This experiment pins MuJoCo 3.1.6; use requirements.txt")
        self.settings = settings or Settings()
        self.model = mujoco.MjModel.from_xml_string(model_xml(self.settings))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = initial_state(seed)
        self.peg_ids = {self.model.geom(f"peg_{j}").id for j in range(2)}
        self.wall_ids = {self.model.geom(f"wall_{j}").id for j in range(3)}
        self.violation_steps = 0
        self.peak_violation = 0.
        self.renderer = None
        self.refresh()

    @property
    def state(self):
        return self.data.qpos.copy()

    def refresh(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def wall_contacts(self):
        self.refresh()
        distances = [float(c.dist) for c in self.data.contact[:self.data.ncon]
                     if ({c.geom1, c.geom2} & self.peg_ids)
                     and ({c.geom1, c.geom2} & self.wall_ids)]
        return len(distances), min(distances, default=1.)

    def step(self, displacement):
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != (9,) or not np.isfinite(displacement).all():
            raise ValueError("Expected nine finite fingertip displacements")
        # Execute finite solver iterates directly; log solver status separately.
        desired = self.data.qpos[7:].copy() + displacement
        mass = np.zeros((15, 15))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        gravity_compensation = mass[6:, 6:] @ np.tile(self.model.opt.gravity, 3)
        for _ in range(self.settings.substeps):
            self.data.ctrl[:] = (100 * (desired - self.data.qpos[7:])
                                 - 2 * self.data.qvel[6:] - gravity_compensation)
            mujoco.mj_step(self.model, self.data)
            q = self.data.qpos[7:]
            violation = float(max(0., np.max(FINGER_LOWER - q), np.max(q - FINGER_UPPER)))
            self.violation_steps += violation > 1e-6
            self.peak_violation = max(self.peak_violation, violation)
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            raise FloatingPointError("MuJoCo warning or non-finite state")

    def frame(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

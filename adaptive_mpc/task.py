"""Task specification; SI units, scalar-first quaternions, body angular velocity."""
from dataclasses import asdict, dataclass

import casadi as ca
import numpy as np


PEG = (([0., 0., 0.], [.040, .012, .012]),
       ([-.015, .020, 0.], [.015, .008, .012]))
WALLS = (([.095, .057, .020], [.095, .020, .020], (0, 2)),
         ([.095, -.025, .020], [.095, .020, .020], (0, 2)),
         ([.195, .008, .020], [.005, .027, .020], (1, 2)))
TARGET = np.array([.105, .008, .012, 1., 0., 0., 0.])
FINGER_LOWER = np.tile([-.18, -.10, .006], 3)
FINGER_UPPER = np.tile([.19, .10, .035], 3)


@dataclass(frozen=True)
class Settings:
    horizon: int = 2
    sharpness: float = 800.
    max_steps: int = 600
    success_hold: int = 10
    model_step: float = .1
    physics_step: float = .002
    substeps: int = 10
    friction: float = .6
    action_limit: float = .005
    solver_iterations: int = 300

    def __post_init__(self):
        if min(self.horizon, self.max_steps, self.success_hold, self.substeps,
               self.solver_iterations) < 1:
            raise ValueError("Step counts must be positive")
        if min(self.sharpness, self.model_step, self.physics_step,
               self.action_limit) <= 0 or self.friction < 0:
            raise ValueError("Invalid physical or numerical setting")

    def to_dict(self):
        return asdict(self)


def initial_state(seed):
    """A fixed peg-local grasp, randomized yaw (+/-24 deg) and lateral offset."""
    rng = np.random.default_rng(4000 + int(seed))
    yaw = np.deg2rad(rng.uniform(-24, 24))
    position = np.array([-.055, .008 + rng.uniform(-.003, .003), .012])
    r2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = np.array([[-.050, 0.], [-.010, .038], [-.010, -.022]]) @ r2.T
    fingers = np.column_stack((xy + position[:2], np.full(3, .012))).ravel()
    return np.r_[position, np.cos(yaw / 2), 0., 0., np.sin(yaw / 2), fingers]


def skew(v):
    return ca.vertcat(ca.horzcat(0, -v[2], v[1]),
                      ca.horzcat(v[2], 0, -v[0]),
                      ca.horzcat(-v[1], v[0], 0))


def rotation(quaternion):
    """Unit-quaternion polynomial, also used at unnormalised predicted states."""
    s = skew(quaternion[1:])
    return ca.DM.eye(3) + 2 * quaternion[0] * s + 2 * s @ s


def velocity_map(q):
    """qdot = B(q) v; v has world translation and body angular velocity."""
    b = ca.MX.zeros(16, 15)
    b[:3, :3] = ca.DM.eye(3)
    b[3:7, 3:6] = .5 * ca.vertcat(-q[4:7].T,
                                  q[3] * ca.DM.eye(3) + skew(q[4:7]))
    b[7:, 6:] = ca.DM.eye(9)
    return b


def running_cost(q, u):
    offsets = [q[i:i+3] - q[:3] for i in (7, 10, 13)]
    local = [rotation(q[3:7]).T @ v for v in offsets]
    directions = [v / ca.norm_2(v) for v in local]
    return (sum(ca.sumsqr(v) for v in offsets)
            + .03 * ca.sumsqr(sum(directions)) + 50 * ca.sumsqr(u))


def terminal_cost(q):
    return 6000 * ca.sumsqr(q[:3] - TARGET[:3]) + 350 * (1 - q[3] ** 2)


def goal_metrics(q, wall_distance):
    position = float(np.linalg.norm(q[:3] - TARGET[:3]))
    orientation = float(1 - np.dot(q[3:7], TARGET[3:7]) ** 2)
    good = q[0] > .090 and position < .012 and orientation < .01 and wall_distance >= -.0005
    return {"position_error_m": position, "orientation_error": orientation,
            "goal_conditions": bool(good)}

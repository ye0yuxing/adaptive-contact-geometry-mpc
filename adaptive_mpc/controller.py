"""Direct multiple-shooting MPC, with one common graph construction for both arms."""
import time

import casadi as ca
import numpy as np

from .dynamics import cf_step
from .geometry import ContactGeometry
from .task import FINGER_LOWER, FINGER_UPPER, Settings, running_cost, terminal_cost


class MPC:
    def __init__(self, mode="adaptive", settings=None):
        if mode not in ("frozen", "adaptive"):
            raise ValueError("mode must be frozen or adaptive")
        self.mode, self.settings = mode, settings or Settings()
        s = self.settings
        self.geometry = ContactGeometry(s.sharpness)
        measured = ca.MX.sym("measured", 16)
        z = ca.MX.sym("trajectory", 25, s.horizon)  # [u_k, q_{k+1}] per column
        geom0 = self.geometry.function(measured)
        states = [measured] + [z[9:, k] for k in range(s.horizon)]
        equalities, objective = [], terminal_cost(states[-1])
        for k in range(s.horizon):
            phi, jn, jt = self.geometry.function(states[k]) if mode == "adaptive" else geom0
            predicted = cf_step(states[k], z[:9, k], phi, jn + s.friction * jt, s.model_step)
            equalities.append(predicted - states[k + 1])
            objective += running_cost(states[k], z[:9, k])
        self.solver = ca.nlpsol("mpc", "ipopt", {
            "x": ca.vec(z), "p": measured, "f": objective, "g": ca.vertcat(*equalities)
        }, {"expand": True, "print_time": False, "error_on_fail": False,
            "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.max_iter": s.solver_iterations,
            "ipopt.hessian_approximation": "limited-memory"})
        lower = np.r_[np.full(9, -s.action_limit), np.full(7, -1e7), FINGER_LOWER]
        upper = np.r_[np.full(9, s.action_limit), np.full(7, 1e7), FINGER_UPPER]
        self.lower, self.upper = np.tile(lower, s.horizon), np.tile(upper, s.horizon)
        self.reset()

    def reset(self):
        self.previous = None

    def action(self, measured):
        started = time.perf_counter()
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (16,) or not np.isfinite(measured).all():
            raise ValueError("Expected 16 finite configuration values")
        if self.previous is None:
            guess = {"x0": np.tile(np.r_[np.zeros(9), measured], self.settings.horizon)}
        else:
            guess = {"x0": self.previous["x"], "lam_x0": self.previous["lam_x"],
                     "lam_g0": self.previous["lam_g"]}
        sol = self.solver(p=measured, lbx=self.lower, ubx=self.upper, lbg=0, ubg=0, **guess)
        elapsed = time.perf_counter() - started
        stats = self.solver.stats()
        trajectory = np.asarray(sol["x"]).reshape(self.settings.horizon, 25)
        if not np.isfinite(trajectory).all():
            raise FloatingPointError("Non-finite MPC iterate; do not hide a solver failure")
        self.previous = sol
        predicted_states = np.vstack((measured, trajectory[:, 9:]))
        phi = [self.geometry.evaluate(q)[0] for q in predicted_states[:-1]]
        change = float(np.linalg.norm(np.asarray(phi) - phi[0])) if self.mode == "adaptive" else 0.
        return trajectory[0, :9].copy(), {
            "solver_status": stats["return_status"],
            "solver_ok": stats["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level"),
            "solver_iterations": int(stats["iter_count"]), "planning_seconds": elapsed,
            "geometry_change": change, "predicted_states": predicted_states,
        }

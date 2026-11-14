"""Explicit dual-cone CF response; formulas credited in docs/provenance.md."""
import casadi as ca

from .task import velocity_map


def cf_step(q, u, gap, jacobian, model_step=.1):
    """Return q_next without an inner solve or additional contact variables.

    Q is a quasi-dynamic control metric, not the MuJoCo physical mass matrix.
    K = I; damping D = 0.1 I is INSIDE SoftPlus.
    Predicted quaternions use first-order integration without renormalization.
    """
    q_inverse = ca.DM([1/50.] * 3 + [1/.06] * 3 + [1/100.] * 9)
    load = ca.vertcat(ca.DM([0., 0., -.098, 0., 0., 0.]), 100 * u)
    free_displacement = q_inverse * load
    contact_motion = jacobian @ free_displacement
    argument = -gap - (1 + .1 / model_step) * contact_motion
    # Stable SoftPlus. This equals log(1 + exp(100*x))/100 mathematically.
    impulse = ca.fmax(argument, 0) + ca.log(1 + ca.exp(-100 * ca.fabs(argument))) / 100
    displacement = free_displacement + q_inverse * (jacobian.T @ impulse)
    return q + velocity_map(q) @ displacement

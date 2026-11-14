import json

import casadi as ca
import numpy as np
import pytest

from adaptive_mpc.controller import MPC
from adaptive_mpc.dynamics import cf_step
from adaptive_mpc.geometry import ContactGeometry
from adaptive_mpc.simulation import Simulation
from adaptive_mpc.task import Settings, TARGET, goal_metrics, initial_state, velocity_map
from analyze import load_trials, summarize


@pytest.fixture(scope="module")
def geometry():
    return ContactGeometry()


@pytest.mark.parametrize("seed", [0, 240, 289])
def test_full_gap_matches_velocity_directional_derivative(geometry, seed):
    rng = np.random.default_rng(seed)
    q = initial_state(seed)
    # Include nonplanar rotations and non-unit predicted quaternions.
    q[3:7] += [.002, .017, -.011, .004]
    v = rng.normal(size=15)
    direction = np.asarray(ca.evalf(velocity_map(ca.DM(q)))) @ v
    epsilon = 1e-7
    plus = geometry.evaluate(q + epsilon * direction)[0]
    minus = geometry.evaluate(q - epsilon * direction)[0]
    _, normal, _ = geometry.evaluate(q)
    np.testing.assert_allclose((plus-minus)/(2*epsilon), normal @ v, rtol=2e-5, atol=2e-7)


def test_geometry_stays_finite_far_from_contact(geometry):
    q = initial_state(0)
    q[:3] += [10, -12, 7]
    for value in geometry.evaluate(q):
        assert np.isfinite(value).all()


def test_cf_response_against_independent_numeric_formula(geometry):
    q, u = initial_state(241), np.linspace(-.004, .004, 9)
    gap, normal, tangent = geometry.evaluate(q)
    j = normal + .6 * tangent
    invq = np.diag([1/50]*3 + [1/.06]*3 + [1/100]*9)
    free = invq @ np.r_[0, 0, -.098, 0, 0, 0, 100*u]
    response = np.logaddexp(0, 100*(-gap-2*j@free)) / 100
    expected = q + np.asarray(ca.evalf(velocity_map(ca.DM(q)))) @ (free + invq @ j.T @ response)
    actual = np.asarray(ca.evalf(cf_step(ca.DM(q), ca.DM(u), ca.DM(gap), ca.DM(j)))).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_horizon_one_has_identical_arms():
    q = initial_state(240)
    frozen = MPC("frozen", Settings(horizon=1))
    adaptive = MPC("adaptive", Settings(horizon=1))
    uf, df = frozen.action(q)
    ua, da = adaptive.action(q)
    assert df["solver_ok"] and da["solver_ok"]
    np.testing.assert_allclose(ua, uf, atol=1e-8, rtol=0)
    assert da["geometry_change"] == df["geometry_change"] == 0


def test_horizon_two_recomputes_geometry():
    q = initial_state(240)
    frozen = MPC("frozen")
    adaptive = MPC("adaptive")
    uf, df = frozen.action(q)
    ua, da = adaptive.action(q)
    assert df["solver_ok"] and da["solver_ok"]
    assert df["geometry_change"] == 0 and da["geometry_change"] > 1e-5
    assert np.linalg.norm(ua-uf) > 1e-6
    assert np.max(np.abs(ua)) <= .005 + 1e-7


def test_simulation_determinism_and_no_mutating_collision_query():
    a, b = Simulation(240), Simulation(240)
    try:
        for _ in range(3):
            a.step(np.zeros(9)); b.step(np.zeros(9))
        np.testing.assert_array_equal(a.state, b.state)
        before = a.state
        a.wall_contacts()
        np.testing.assert_array_equal(before, a.state)
        assert np.isfinite(a.state).all()
    finally:
        a.close(); b.close()


def test_success_requires_pose_and_physical_wall_clearance():
    q = initial_state(240)
    q[:7] = TARGET
    assert goal_metrics(q, 0.)["goal_conditions"]
    assert not goal_metrics(q, -.00051)["goal_conditions"]
    q[0] = .089
    assert not goal_metrics(q, 0.)["goal_conditions"]


def test_paired_analysis_includes_failures_and_rejects_missing_pairs():
    rows = [{"seed": seed, "mode": mode, "success": mode == "adaptive",
             "steps": 10, "solver_success_fraction": .8,
             "median_planning_seconds": .01}
            for seed in range(4) for mode in ("frozen", "adaptive")]
    report = summarize(rows)
    assert report["paired_outcomes"]["rescues"] == 4
    assert report["paired_outcomes"]["exact_two_sided_p"] == .125
    assert report["arms"]["frozen"]["median_successful_steps"] is None
    with pytest.raises(ValueError):
        summarize(rows[:-1])
    with pytest.raises(ValueError):
        summarize(rows + rows[:1])


def test_analysis_loads_only_complete_runner_output(tmp_path):
    rows = [{"seed": 0, "mode": mode, "success": False,
             "steps": 5, "solver_success_fraction": 1.,
             "median_planning_seconds": .01} for mode in ("frozen", "adaptive")]
    data = {"seeds": [0], "modes": ["frozen", "adaptive"], "runs": rows}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data))
    assert summarize(load_trials(path))["pairs"] == 1
    with pytest.raises(ValueError):
        load_trials(tmp_path / "progress.json")
    data["seeds"] = [0, 1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_trials(path)

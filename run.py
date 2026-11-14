"""Run an explicitly configured frozen/adaptive comparison; no network access."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Set before importing NumPy/CasADi; users may override them explicitly.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import casadi
import mujoco
import numpy as np

from adaptive_mpc.controller import MPC
from adaptive_mpc.simulation import Simulation
from adaptive_mpc.task import Settings, goal_metrics, initial_state


def source_hashes():
    root = Path(__file__).resolve().parent
    files = [root / "run.py", *sorted((root / "adaptive_mpc").glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def rollout(seed, mode, settings, destination):
    simulation = Simulation(seed, settings)
    controller = MPC(mode, settings)
    records, states, actions = [], [simulation.state], []
    hold = 0
    try:
        for step in range(settings.max_steps):
            simulation.refresh()
            action, diagnostic = controller.action(simulation.state)
            simulation.step(action)
            count, distance = simulation.wall_contacts()
            q = simulation.state
            metrics = goal_metrics(q, distance)
            hold = hold + 1 if metrics["goal_conditions"] else 0
            diagnostic.pop("predicted_states")
            records.append({"seed": seed, "mode": mode, "step": step, **diagnostic,
                            **metrics, "peg_x_m": float(q[0]), "wall_contacts": count,
                            "wall_distance_m": distance,
                            "action_norm_m": float(np.linalg.norm(action))})
            states.append(q)
            actions.append(action)
            if hold >= settings.success_hold:
                break
    finally:
        simulation.close()
    name = f"{mode}_{seed}"
    with (destination / f"{name}.csv").open("w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(records[0]))
        writer_csv.writeheader()
        writer_csv.writerows(records)
    np.savez_compressed(destination / f"{name}.npz", states=states, actions=actions)
    return {"seed": seed, "mode": mode, "success": hold >= settings.success_hold,
            "steps": len(records), "initial_state": initial_state(seed).tolist(),
            "position_error_m": records[-1]["position_error_m"],
            "orientation_error": records[-1]["orientation_error"],
            "solver_success_fraction": float(np.mean([r["solver_ok"] for r in records])),
            "median_planning_seconds": float(np.median([r["planning_seconds"] for r in records])),
            "minimum_wall_distance_m": min(r["wall_distance_m"] for r in records),
            "wall_contact_steps": sum(r["wall_contacts"] > 0 for r in records),
            "workspace_violation_physics_steps": int(simulation.violation_steps),
            "peak_workspace_violation_m": simulation.peak_violation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[240])
    parser.add_argument("--seed-range", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--modes", choices=["frozen", "adaptive"], nargs="+", default=["frozen", "adaptive"])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--sharpness", type=float, default=800.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = list(range(*args.seed_range)) if args.seed_range else args.seeds
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Use nonempty, unique, nonnegative seeds and unique modes")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings(horizon=args.horizon, max_steps=args.max_steps, sharpness=args.sharpness)
    metadata = {"settings": settings.to_dict(), "seeds": seeds, "modes": args.modes,
                "source_sha256": source_hashes(), "versions": {"python": platform.python_version(),
                "numpy": np.__version__, "mujoco": mujoco.__version__, "casadi": casadi.__version__}}
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.perf_counter()
    rows = []
    for seed in seeds:
        for mode in args.modes:
            result = rollout(seed, mode, settings, args.output)
            rows.append(result)
            (args.output / "progress.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
            print(f"{mode:8s} seed={seed} success={result['success']} steps={result['steps']}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(
        {**metadata, "elapsed_seconds": time.perf_counter()-started, "runs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

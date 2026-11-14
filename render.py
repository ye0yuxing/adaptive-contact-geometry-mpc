"""Render a saved paired rollout as a GIF; no controller rerun is required."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adaptive_mpc.simulation import Simulation
from adaptive_mpc.task import Settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, default=240)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.stride < 1 or args.output.exists() or args.output.suffix.lower() != ".gif":
        p.error("Use a positive stride and a new .gif output path")
    protocol = json.loads((args.directory / "protocol.json").read_text())
    settings = Settings(**protocol["settings"])
    modes = ("frozen", "adaptive")
    trajectories = []
    for mode in modes:
        with np.load(args.directory / f"{mode}_{args.seed}.npz", allow_pickle=False) as data:
            states = data["states"]
        if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all():
            raise ValueError("Invalid state trajectory")
        trajectories.append(states)
    simulation, frames = Simulation(args.seed, settings), []
    # Rendering-only camera adjustment; it has no effect on the saved rollout.
    simulation.model.cam_pos[0] = [.04, -.09, .30]
    mujoco.mju_axisAngle2Quat(simulation.model.cam_quat[0], np.array([1., 0., 0.]),
                            np.arctan2(.10, .30))
    font = ImageFont.load_default(size=20)
    length = max(len(q) for q in trajectories)
    try:
        indices = sorted(set([*range(0, length, args.stride), length - 1]))
        for step in indices:
            canvas = Image.new("RGB", (1280, 520), "#15202b")
            for column, (mode, states) in enumerate(zip(modes, trajectories)):
                index = min(step, len(states) - 1)
                simulation.data.qpos[:] = states[index]
                mujoco.mj_forward(simulation.model, simulation.data)
                canvas.paste(Image.fromarray(simulation.frame()), (column * 640, 40))
                ImageDraw.Draw(canvas).text((column * 640 + 18, 12),
                    f"{mode.upper()} | seed {args.seed} | step {index}", fill="white", font=font)
            frames.append(canvas.quantize(colors=128))
    finally:
        simulation.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = round(1000 * settings.physics_step * settings.substeps * args.stride)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)
    print(f"Saved {len(frames)} frames to {args.output.name}; aligned by simulation step")


if __name__ == "__main__":
    main()

"""Summarize paired trials without discarding failed runs or solver statuses."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    center = (rate + z*z / (2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., center-radius), min(1., center+radius)]


def summarize(rows):
    pairs = {}
    for row in rows:
        mode, seed = row["mode"], row["seed"]
        if mode not in ("frozen", "adaptive") or not isinstance(seed, int) or seed < 0:
            raise ValueError("Invalid mode or seed")
        if not isinstance(row["success"], bool):
            raise ValueError("Success must be a boolean")
        if mode in pairs.setdefault(seed, {}):
            raise ValueError(f"Duplicate run for seed {seed}")
        if row["steps"] < 1 or not 0 <= row["solver_success_fraction"] <= 1:
            raise ValueError("Invalid rollout length or solver success fraction")
        if not math.isfinite(row["median_planning_seconds"]) or row["median_planning_seconds"] < 0:
            raise ValueError("Invalid planning time")
        pairs[seed][mode] = row
    if not pairs or any(set(p) != {"frozen", "adaptive"} for p in pairs.values()):
        raise ValueError("Expected complete frozen/adaptive pairs; no filtering is permitted")
    result = {"paired_seeds": sorted(pairs), "pairs": len(pairs), "arms": {}}
    for mode in ("frozen", "adaptive"):
        runs = [pair[mode] for pair in pairs.values()]
        n = sum(r["success"] for r in runs)
        lengths = [r["steps"] for r in runs if r["success"]]
        result["arms"][mode] = {
            "successes": n, "trials": len(runs), "success_rate": n / len(runs),
            "wilson_95": wilson(n, len(runs)),
            "median_successful_steps": median(lengths) if lengths else None,
            "median_per_run_planning_seconds": median(r["median_planning_seconds"] for r in runs),
            "median_solver_success_fraction": median(r["solver_success_fraction"] for r in runs),
            "minimum_solver_success_fraction": min(r["solver_success_fraction"] for r in runs),
            "failure_seeds": sorted(r["seed"] for r in runs if not r["success"]),
        }
    outcomes = {"rescues": 0, "harms": 0, "both_success": 0, "both_failure": 0}
    for pair in pairs.values():
        f, a = pair["frozen"]["success"], pair["adaptive"]["success"]
        label = "both_success" if f and a else "harms" if f else "rescues" if a else "both_failure"
        outcomes[label] += 1
    discordant = outcomes["rescues"] + outcomes["harms"]
    tail = min(outcomes["rescues"], outcomes["harms"])
    probability = min(1., 2 * sum(math.comb(discordant, k) for k in range(tail+1)) / 2**discordant)
    result["paired_outcomes"] = {**outcomes, "exact_two_sided_p": probability}
    return result


def load_trials(path):
    if path.name != "summary.json":
        raise ValueError("Use the completed summary.json produced by run.py")
    data = json.loads(path.read_text())
    rows = data["runs"]
    actual = [(r["seed"], r["mode"]) for r in rows]
    expected = {(seed, mode) for seed in data["seeds"] for mode in data["modes"]}
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Trial coverage does not match the declared protocol")
    return rows


def plot(report, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    arms = [report["arms"][mode] for mode in ("frozen", "adaptive")]
    rates = [100 * arm["success_rate"] for arm in arms]
    intervals = [[rates[i]-100*a["wilson_95"][0] for i, a in enumerate(arms)],
                 [100*a["wilson_95"][1]-rates[i] for i, a in enumerate(arms)]]
    ax.bar([0, 1], rates, color=["#7e8b9b", "#168578"], width=.55,
           yerr=intervals, capsize=6)
    ax.set(xticks=[0, 1], xticklabels=["Frozen geometry", "Adaptive geometry"],
           ylim=(0, 115), ylabel="Insertion success (%)", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    for i, arm in enumerate(arms):
        ax.text(i, 103, f'{arm["successes"]}/{arm["trials"]}', ha="center", fontweight="bold")
    ax.set_xlabel("Paired initial conditions; 95% Wilson intervals", labelpad=12, fontsize=9)
    fig.savefig(output, dpi=180, metadata={"Software": "adaptive-contact-geometry-mpc"})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--plot", type=Path)
    p.add_argument("--title", default="Frozen vs. adaptive contact geometry")
    args = p.parse_args()
    for destination in (args.output, args.plot):
        if destination and destination.exists():
            p.error("Output already exists; choose a new path")
    report = summarize(load_trials(args.input))
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    if args.plot:
        plot(report, args.plot, args.title)


if __name__ == "__main__":
    main()

"""Compare occupancy intervals for five counters with two unknown faults."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

import numpy as np

from fault_telemetry import (
    audit, banks as earlier_fault_banks, conditional, interval_hull,
    observations, with_empty_sentinels,
)
from telemetry import (
    GATES, MENU, PUBLISHED, TOL, SensorModel, drift_rows, generate_fields,
    rounded, trajectory,
)


PROTOCOL_SHA256 = "48fd0bb23dea13dcd7314bf9ca054dccc8e7dac8b15968adea665c385d13d291"
SPLITS = {
    "nominal": (16708821, 5509181, 64, tuple(range(1, 8))),
    "shift": (16709113, 5509543, 24, (1, 2, 4, 8, 16)),
}
COUNTER_SETS = tuple(combinations(range(len(MENU)), 5))
PROBES_PER_PAIR = 4
BOOTSTRAP_SEED = 7848912
TOLERANCE = 8 * TOL


def new_banks():
    used = set(PUBLISHED)
    for seed, count, values in (
        (848322, 48, tuple(range(1, 8))),
        (837612, 128, tuple(range(1, 8))),
        (882157, 64, (1, 2, 4, 8, 16)),
    ):
        generate_fields(seed, count, values, used)
    for previous_fields, _, _ in earlier_fault_banks().values():
        used.update(previous_fields)
    return {
        split: (generate_fields(field_seed, count, values, used), tape_seed)
        for split, (field_seed, tape_seed, count, values) in SPLITS.items()
    }


def honest_tape(rates, kernels, rng):
    initial = np.array([rng.uniform(-.12, .12) for _ in range(2)])
    step = rng.choice((.04, .10))
    horizon = rng.choice((24, 40, 64))
    width = step * rng.choice((.5, 1.5, 2.5))
    lag = (rng.randint(1, 3), rng.randint(1, 3))
    drift_radius = np.array([.018 * (2 ** rng.uniform(0., 3.)) for _ in range(2)])
    gate_radius = np.array([.004 * (2 ** rng.uniform(0., 4.)) for _ in MENU])
    outcomes = [trajectory(rates, initial, step, horizon, width, kernel, lag) for kernel in kernels]
    occupancy = np.array([value[0] for value in outcomes])
    drift = np.array([value[1] for value in outcomes])
    noise = np.array([[rng.uniform(-1, 1) for _ in range(2 + len(MENU))] for _ in kernels])
    return occupancy, drift, noise, drift_radius, gate_radius


def diameter(model, drift_radius, gate_radius, counter):
    return model.certified_diameter(np.r_[drift_radius, gate_radius[counter]])


def choose_counters(models, drift_radius, gate_radius):
    widths = {
        counter: diameter(models[(counter,)], drift_radius, gate_radius, counter)
        for counter in range(len(MENU))
    }
    scores = {subset: max(widths[counter] for counter in subset) for subset in COUNTER_SETS}
    selected = min(COUNTER_SETS, key=lambda subset: round(scores[subset], 11))
    return selected, scores[selected]


def far_alternative(model, readings, radii, truth):
    centered = readings - model.offset
    simplex_limits = np.tile([0., 0., 0., 1.], (len(truth), 1))
    rhs = np.hstack((simplex_limits, centered + radii, -centered + radii))
    vertices, feasible = model.conditional.feasible(rhs)
    assert feasible.any(axis=1).all()
    joint_duty = 1. - vertices.sum(axis=2)
    distance = np.round(np.abs(joint_duty - truth[:, None]), 12)
    choice = np.argmax(np.where(feasible, distance, -np.inf), axis=1)
    reduced = vertices[np.arange(len(truth)), choice]
    return np.column_stack((reduced, 1. - reduced.sum(axis=1)))


def append_probes(models, selected, drift, gates, drift_radius, gate_radius, truth):
    probes = [gates]
    alternative_targets = []
    for pair in combinations(selected, 2):
        shared = next(counter for counter in selected if counter not in pair)
        alternative = far_alternative(
            models[(shared,)],
            observations(drift, gates, (shared,)),
            np.r_[drift_radius, gate_radius[shared]],
            truth,
        )
        columns = list(pair)
        values = gates[:, columns]
        fault_span = 4 * gate_radius[columns]
        replacements = [
            np.clip(values - fault_span, 0., 1.),
            np.clip(values + fault_span, 0., 1.),
            np.full_like(values, .5),
            alternative @ GATES[columns].T,
        ]
        for replacement in replacements:
            report = gates.copy()
            report[:, columns] = replacement
            probes.append(report)
        alternative_targets.append(alternative[:, 3])
    return np.stack(probes), np.stack(alternative_targets)


def bounded(model, support, drift, gates, drift_radius, gate_radius):
    readings = observations(drift, gates, support)
    radii = np.r_[drift_radius, gate_radius[list(support)]]
    interval, valid = conditional(model, readings, radii)
    return with_empty_sentinels(interval, valid)


def exact_interval(models, supports, drift, tape, drift_radius, gate_radius):
    return interval_hull(np.stack([
        bounded(models[support], support, drift, tape, drift_radius, gate_radius)
        for support in supports
    ]))


def relaxation(models, supports, drift, tape, drift_radius, gate_radius):
    pair_bounds = {
        pair: bounded(models[pair], pair, drift, tape, drift_radius, gate_radius)
        for pair in sorted({pair for support in supports for pair in combinations(support, 2)})
    }
    projected = []
    for support in supports:
        intervals = np.stack([pair_bounds[pair] for pair in combinations(support, 2)])
        lo = intervals[..., 0].max(axis=0)
        hi = intervals[..., 1].min(axis=0)
        projected.append(with_empty_sentinels(np.stack((lo, hi), axis=-1), lo <= hi + TOL))
    return interval_hull(np.stack(projected))


def check_nested(inner, outer):
    assert np.all(inner[..., 0] >= outer[..., 0] - TOLERANCE)
    assert np.all(inner[..., 1] <= outer[..., 1] + TOLERANCE)


def check_coherent(interval, alternative, truth):
    coherent = interval[PROBES_PER_PAIR::PROBES_PER_PAIR]
    low = np.minimum(alternative, truth)
    high = np.maximum(alternative, truth)
    assert (coherent[..., 0] <= low + TOLERANCE).all()
    assert (coherent[..., 1] >= high - TOLERANCE).all()


def metrics(interval, truth):
    counts = audit(interval, truth)
    width = interval[..., 1] - interval[..., 0]
    valid = np.isfinite(width)
    assert ((width[valid] >= -TOL) & (width[valid] <= 1 + TOL)).all()
    width = np.where(valid, np.maximum(width, 0.), 0.)
    whole = valid[1:].all(axis=0)
    return counts | {
        "nonempty_uncovered": counts["total"] - counts["covered"] - counts["empty"],
        "clean_width_sum": float(width[0].sum()),
        "clean_valid": int(valid[0].sum()),
        "clean_covered": audit(interval[:1], truth)["covered"],
        "fault_width_sum": float(width[1:].sum()),
        "fault_valid": int(valid[1:].sum()),
        "worst_fault_width_sum": float(width[1:, whole].max(axis=0).sum()),
        "worst_fault_trajectories": int(whole.sum()),
        "correct": counts["resolved"] - counts["wrong"],
    }


def totals(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def summarise(sums):
    result = {key: value for key, value in sums.items() if not key.endswith("_width_sum")}
    for index, denominator in (
        ("clean_width", "clean_valid"),
        ("fault_width", "fault_valid"),
        ("worst_fault_width", "worst_fault_trajectories"),
    ):
        result[index] = sums[f"{index}_sum"] / sums[denominator] if sums[denominator] else None
    return result


def setting(models, rates, kernels, rng, shifted):
    occupancy, exact_drift, noise, dr, gr = honest_tape(rates, kernels, rng)
    truth = occupancy[:, 3]
    nominal_drift = exact_drift + noise[:, :2] * dr
    gates = occupancy @ GATES.T + noise[:, 2:] * gr
    selected, bound = choose_counters(models, dr, gr)
    tape, alternative = append_probes(models, selected, nominal_drift, gates, dr, gr, truth)
    triple = tuple(combinations(selected, 3))
    if shifted:
        drift = exact_drift + 3 * noise[:, :2] * dr
        correct_radii = exact_interval(models, triple, drift, tape, 3 * dr, gr)
        understated = exact_interval(models, triple, drift, tape, dr, gr)
        widened_bound = max(diameter(models[(i,)], 3 * dr, gr, i) for i in selected)
        check_nested(understated, correct_radii)
        assert np.max(correct_radii[..., 1] - correct_radii[..., 0]) <= widened_bound + TOLERANCE
        check_coherent(correct_radii, alternative, truth)
        outcomes = {"correct_shift_radius": correct_radii, "understated_shift_radius": understated}
        certified = ("correct_shift_radius",)
    else:
        exact = exact_interval(models, triple, nominal_drift, tape, dr, gr)
        envelope = relaxation(models, triple, nominal_drift, tape, dr, gr)
        single = np.stack([
            bounded(models[(i,)], (i,), nominal_drift, tape, dr, gr) for i in selected
        ])
        median = np.median(single, axis=0)
        one_fault = exact_interval(
            models, tuple(combinations(selected, 4)), nominal_drift, tape, dr, gr
        )
        check_nested(exact, envelope)
        check_nested(exact, median)
        check_nested(one_fault, exact)
        assert np.max(exact[..., 1] - exact[..., 0]) <= bound + TOLERANCE
        check_coherent(exact, alternative, truth)
        outcomes = {"exact": exact, "pairwise_envelope": envelope, "median": median,
                    "one_fault_control": one_fault}
        certified = ("exact", "pairwise_envelope", "median")
    measured = {name: metrics(interval, truth) for name, interval in outcomes.items()}
    for name in certified:
        summary = measured[name]
        assert summary["covered"] == summary["total"] and summary["wrong"] == 0
    return measured, np.abs(alternative - truth)


def evaluate(fields, seed, shifted):
    rng = Random(seed)
    kernels = ("delay" if shifted else "sign", "hysteresis", "smooth")
    holdout = []
    gaps = []
    for field in fields:
        rates = drift_rows(field)
        sizes = (1, 3) if shifted else (1, 2, 3, 4)
        models = {
            support: SensorModel(np.vstack((rates, GATES[list(support)])))
            for length in sizes for support in combinations(range(len(MENU)), length)
        }
        settings = []
        for _ in range(2):
            outcome, alternatives = setting(models, rates, kernels, rng, shifted)
            settings.append(outcome)
            gaps.extend(alternatives.ravel().tolist())
        holdout.append({name: totals([entry[name] for entry in settings]) for name in outcome})
    report = {
        "fields": len(fields),
        "trajectories": 6 * len(fields),
        "coherent_probes": len(gaps),
        "coherent_alternative_mean_separation": float(np.mean(gaps)),
        "coherent_alternative_max_separation": float(np.max(gaps)),
        "means": {
            name: summarise(totals([row[name] for row in holdout]))
            for name in holdout[0]
        },
        "per_field": [
            {"field": list(field), "methods": {name: summarise(values) for name, values in row.items()}}
            for field, row in zip(fields, holdout)
        ],
    }
    return report


def paired_comparison(primary):
    comparison = np.array([
        [row["methods"][name]["worst_fault_width"] for name in ("pairwise_envelope", "exact")]
        for row in primary["per_field"]
    ])
    paired = comparison[:, 0] - comparison[:, 1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(paired), size=(4000, len(paired)))
    interval = np.quantile(paired[draws].mean(axis=1), [.025, .975]).tolist()
    mean = float(paired.mean())
    relative = mean / float(comparison[:, 0].mean())
    return {
        "mean_reduction": mean, "relative_reduction": relative,
        "paired_95_interval": interval,
        "practical_gate_passed": bool(mean >= .005 and relative >= .10 and interval[0] > 0),
    }


def study():
    protocol = Path(__file__).with_name("TWO_FAULT_PROTOCOL.md").read_bytes()
    if sha256(protocol).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("The frozen protocol hash does not match.")
    reports = {
        name: evaluate(fields, seed, shifted=name == "shift")
        for name, (fields, seed) in new_banks().items()
    }
    return rounded({
        "protocol_sha256": PROTOCOL_SHA256,
        "split_specification": SPLITS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_comparison": paired_comparison(reports["nominal"]),
        "cohorts": reports,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = json.dumps(study(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    elif result != args.verify.read_text():
        raise ValueError("The two-fault telemetry replay differed from the saved results.")
    else:
        print("Two-fault telemetry replay matches exactly.")


if __name__ == "__main__":
    main()

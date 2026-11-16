"""Partial identification with one unknown faulty counter."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from random import Random

import numpy as np

from telemetry import (
    DIFFERENCE, GATES, MENU, PUBLISHED, THRESHOLDS, TOL, SensorModel,
    drift_rows, generate_fields, trajectory,
)


PROTOCOL = "f196e67abc5789b8a8c32d28e41d97e8aa43f6e9a7ec9ce6149e83fe1ec64da6"
PAIRS = tuple(combinations(range(len(GATES)), 2))
TRIPLETS = tuple(combinations(range(len(GATES)), 3))
FAULT_COUNT = 4
PROBE_COUNT = 1 + FAULT_COUNT * len(GATES)


def probe_indices(counter):
    start = 1 + FAULT_COUNT * counter
    return np.arange(start, start + FAULT_COUNT)


def conditional(model, readings, radii):
    shape = readings.shape[:-1]
    bounds, valid = model.intervals(readings.reshape(-1, readings.shape[-1]), radii)
    return bounds.reshape((*shape, 2)), valid.reshape(shape)


def with_empty_sentinels(bounds, valid):
    lower = np.where(valid, bounds[..., 0], np.inf)
    upper = np.where(valid, bounds[..., 1], -np.inf)
    return np.stack((lower, upper), axis=-1)


def interval_hull(pair_bounds):
    return np.stack((pair_bounds[..., 0].min(axis=0), pair_bounds[..., 1].max(axis=0)), axis=-1)


def attaining_states(model, radii):
    """Lift a diameter-attaining difference from the three reduced coordinates."""
    budgets = np.concatenate((np.ones(len(DIFFERENCE)), 2 * radii, 2 * radii))
    candidates, valid = model.differences.feasible(budgets)
    objectives = np.where(valid[0], -candidates[0].sum(axis=1), -np.inf)
    idx = int(np.argmax(objectives))
    delta = np.append(candidates[0, idx], objectives[idx])
    first, second = np.maximum(delta, 0), np.maximum(-delta, 0)
    shared_slack = (1 - first.sum()) / 4
    return first + shared_slack, second + shared_slack


def audit_sharpness(rates, drift_radius, gate_radius, triplet, diameter, singles):
    shared = max(triplet, key=diameter.__getitem__)
    others = [counter for counter in triplet if counter != shared]
    radii = np.append(drift_radius, gate_radius[shared])
    first, second = attaining_states(singles[shared], radii)
    mean = (first + second) * .5
    readings = {shared: GATES[shared] @ mean, others[0]: GATES[others[0]] @ first,
                others[1]: GATES[others[1]] @ second}
    assert np.all(first >= -TOL) and np.all(second >= -TOL)
    assert abs(first.sum() - 1) <= TOL and abs(second.sum() - 1) <= TOL
    fixed = rates @ mean
    resolved = []
    for pair in combinations(triplet, 2):
        model = SensorModel(np.vstack((rates, GATES[list(pair)])))
        bounds, valid = model.intervals(
            np.r_[fixed, [readings[counter] for counter in pair]],
            np.r_[drift_radius, gate_radius[list(pair)]],
        )
        if valid[0]:
            resolved.append(bounds[0])
    hull = np.array([min(x[0] for x in resolved), max(x[1] for x in resolved)])
    assert abs(hull[0] - second[3]) <= 8 * TOL
    assert abs(hull[1] - first[3]) <= 8 * TOL
    assert abs(hull[1] - hull[0] - diameter[shared]) <= 8 * TOL


def rational_audit():
    # The sound marginals both equal 1/2.  Only t = p11 is unknown.
    grid = [F(i, 16) for i in range(9)]
    radii = [F(1, 8), F(1, 4), F(1, 16)]
    code = lambda t: (t, F(1, 2) - t, 1 - 2 * t)
    final = F(0), F(1, 4), F(0)
    compatible = lambda t: sum(abs(a - b) <= r for a, b, r in zip(code(t), final, radii)) >= 2
    assert compatible(F(0)) and compatible(F(1, 2))
    audited, rejected, max_width = 0, 0, F(0)
    for calibrations in (radii, [F(1, 32)] * 3):
        for p, q in combinations(grid, 2):
            a, b = code(p), code(q)
            overlap = any(abs(lo - hi) <= 2 * r for lo, hi, r in zip(a, b, calibrations))
            reports = [(lo, hi, (lo + hi) / 2) for lo, hi in zip(a, b)]
            good = lambda vals, z: sum(abs(x - y) <= r for x, y, r in zip(vals, z, calibrations)) >= 2
            share = any(good(a, z) and good(b, z) for z in product(*reports))
            assert share == overlap
            audited += 1
            rejected += not share
            if share:
                max_width = max(max_width, q - p)
    assert max_width == F(1, 2) and rejected > 0
    return {"x11": [str(grid[0]), str(grid[-1])], "shared_readings": list(map(str, final)),
            "audited_grid_pairs": audited, "nonoverlapping_grid_pairs": rejected}


def tapes(rates, kernels, rng, shift):
    initial = np.array([rng.uniform(-.12, .12) for _ in range(2)])
    step = rng.choice((.04, .10))
    horizon = rng.choice((24, 40, 64))
    width = step * rng.choice((.5, 1.5, 2.5))
    lag = (rng.randint(1, 3), rng.randint(1, 3))
    drift_radius = np.array([.018 * (2 ** rng.uniform(0., 3.)) for _ in range(2)])
    if shift:
        drift_radius *= 3
    gate_radius = np.array([.004 * (2 ** rng.uniform(0., 4.)) for _ in MENU])
    outcomes = [trajectory(rates, initial, step, horizon, width, kernel, lag) for kernel in kernels]
    occupancy = np.array([row[0] for row in outcomes])
    exact_drift = np.array([row[1] for row in outcomes])
    noise = np.array([[rng.uniform(-1, 1) for _ in range(2 + len(MENU))] for _ in kernels])
    displacements = exact_drift + noise[:, :2] * drift_radius
    nominal = occupancy @ GATES.T + noise[:, 2:] * gate_radius
    alone = SensorModel(rates)
    center = displacements - alone.offset
    ref = np.tile([0., 0., 0., 1.], (len(kernels), 1))
    coordinates, valid = alone.conditional.feasible(np.hstack((ref, center + drift_radius, -center + drift_radius)))
    assert valid.any(axis=1).all()
    probabilities = np.concatenate((coordinates, 1 - coordinates.sum(axis=2, keepdims=True)), axis=2)
    proposals = probabilities @ GATES.T
    severity = np.where(valid[..., None], np.abs(proposals - nominal[:, None, :]), -np.inf)
    far = np.take_along_axis(proposals, severity.argmax(axis=1)[:, None, :], axis=1)[:, 0, :]
    perturbed = np.tile(nominal, (PROBE_COUNT, 1, 1))
    for i in range(len(MENU)):
        perturbed[probe_indices(i), :, i] = np.array([
            np.clip(nominal[:, i] - 4 * gate_radius[i], 0, 1),
            np.clip(nominal[:, i] + 4 * gate_radius[i], 0, 1),
            np.full(len(kernels), .5),
            np.clip(far[:, i], 0, 1),
        ])
    return occupancy[:, 3], displacements, perturbed, drift_radius, gate_radius


def observations(drift, gates, counters):
    repeat = np.broadcast_to(drift, (*gates.shape[:-1], 2))
    return np.concatenate((repeat, gates[..., list(counters)]), axis=-1)


def audit(intervals, truth):
    lo, hi = intervals[..., 0], intervals[..., 1]
    valid = lo <= hi + TOL
    contained = valid & (lo <= truth + TOL) & (hi >= truth - TOL)
    above = valid[..., None] & (lo[..., None] > THRESHOLDS + TOL)
    below = valid[..., None] & (hi[..., None] < THRESHOLDS - TOL)
    wrong = (above & (truth[:, None] <= THRESHOLDS)) | (below & (truth[:, None] >= THRESHOLDS))
    return {
        "covered": int(contained.sum()), "empty": int((~valid).sum()), "total": int(valid.size),
        "resolved": int((above | below).sum()), "wrong": int(wrong.sum()),
    }


def interval_metrics(intervals):
    widths = intervals[..., 1] - intervals[..., 0]
    return {
        "worst_probe_width": float(widths[1:].max(axis=0).mean()),
        "mean_probe_width": float(widths[1:].mean()),
        "clean_width": float(widths[0].mean()),
    }


def collect_intervals(models, displacements, gates, drift_radius, counters_radius):
    intervals = {}
    for counters, model in models.items():
        probes = np.concatenate(([0], *(probe_indices(i) for i in counters)))
        points = observations(displacements, gates[probes], counters)
        radii = np.r_[drift_radius, counters_radius[list(counters)]]
        bounds, valid = conditional(model, points, radii)
        result = np.tile(with_empty_sentinels(bounds[0], valid[0]), (PROBE_COUNT, 1, 1))
        result[probes] = with_empty_sentinels(bounds, valid)
        intervals[counters] = result
    return intervals


def scenario(rates, kernels, rng, shift, single_models, pair_models, geometry):
    truth, drift, gates, dr, gr = tapes(rates, kernels, rng, shift)
    bank = collect_intervals(single_models | pair_models, drift, gates, dr, gr)
    one = [single_models[(i,)] for i in range(len(MENU))]
    diameters = [model.certified_diameter(np.r_[dr, gr[i]]) for i, model in enumerate(one)]
    scores = {triplet: max(diameters[i] for i in triplet) for triplet in TRIPLETS}
    selected = min(TRIPLETS, key=lambda tri: round(scores[tri], 11))
    projection = sorted(range(len(MENU)), key=lambda i: (-abs(geometry[i]) / gr[i], i))
    precision = sorted(range(len(MENU)), key=lambda i: (gr[i], i))
    choices = {"minimax": selected, "geometry": tuple(sorted(projection[:3])),
               "precision": tuple(sorted(precision[:3]))}
    audit_sharpness(rates, dr, gr, selected, diameters, one)
    metrics = {}
    for triplet in TRIPLETS:
        probes = np.concatenate(([0], *(probe_indices(i) for i in triplet)))
        pairs = tuple(combinations(triplet, 2))
        candidates = np.stack([bank[p][probes] for p in pairs])
        exact = interval_hull(candidates)
        checked = audit(exact, truth)
        assert checked["covered"] == checked["total"] and checked["wrong"] == 0
        assert np.max(exact[..., 1] - exact[..., 0]) <= scores[triplet] + 8 * TOL
        metrics[triplet] = interval_metrics(exact) | checked | {"diameter": scores[triplet]}
        if triplet != selected:
            continue
        independent = np.stack([bank[(i,)][probes] for i in triplet])
        median = np.median(independent, axis=0)
        baseline = audit(median, truth)
        assert baseline["covered"] == baseline["total"] and baseline["wrong"] == 0
        assert np.all(exact[..., 0] >= median[..., 0] - 8 * TOL)
        assert np.all(exact[..., 1] <= median[..., 1] + 8 * TOL)
        metrics["median"] = interval_metrics(median) | baseline
        oracle = np.concatenate([bank[tuple(j for j in triplet if j != i)][probe_indices(i)]
                                 for i in triplet])
        fragile = SensorModel(np.vstack((rates, GATES[list(triplet)])))
        bounds, okay = conditional(fragile, observations(drift, gates[probes], triplet), np.r_[dr, gr[list(triplet)]])
        observed = with_empty_sentinels(bounds, okay)
        diagnosed = np.concatenate((observed[:1], oracle))
        oracle_audit = audit(diagnosed, truth)
        assert oracle_audit["covered"] == oracle_audit["total"] and oracle_audit["wrong"] == 0
        metrics["oracle"] = interval_metrics(diagnosed) | oracle_audit
        metrics["all_three"] = audit(observed, truth)
    return choices, metrics


def banks():
    used = set(PUBLISHED)
    return {
        "development": (generate_fields(920159, 32, range(1, 8), used), 124203, False),
        "primary": (generate_fields(920173, 96, range(1, 8), used), 124211, False),
        "shift": (generate_fields(920189, 48, [1, 2, 4, 8, 16], used), 124223, True),
    }


def evaluate(states, seed, shift):
    rng = Random(seed)
    bank = []
    kernels = ("delay" if shift else "sign", "hysteresis", "smooth")
    for field in states:
        rates = drift_rows(field)
        single = {(i,): SensorModel(np.vstack((rates, GATES[i]))) for i in range(len(MENU))}
        pair = {ij: SensorModel(np.vstack((rates, GATES[list(ij)]))) for ij in PAIRS}
        _, _, right = np.linalg.svd(np.vstack((np.ones(4), rates)))
        geometry = GATES @ right[-1]
        bank.append([scenario(rates, kernels, rng, shift, single, pair, geometry) for _ in range(2)])
    return bank


def average(rows):
    result = {}
    for key in rows[0]:
        reducer = np.sum if key in {"covered", "empty", "total", "resolved", "wrong"} else np.mean
        value = reducer([row[key] for row in rows])
        result[key] = int(value) if isinstance(value, np.integer) else float(value)
    return result


def field_metrics(bank, fixed):
    result = []
    for settings in bank:
        records = []
        for choices, data in settings:
            records.append({name: data[key] for name, key in (choices | {"development_fixed": fixed}).items()}
                           | {name: data[name] for name in ("median", "oracle", "all_three")})
        result.append({name: average([row[name] for row in records]) for name in records[0]})
    return result


def bootstrap(field_rows):
    paired = np.array([field["median"]["worst_probe_width"] - field["minimax"]["worst_probe_width"]
                       for field in field_rows])
    rng = np.random.default_rng(308247)
    sample = rng.integers(len(paired), size=(4000, len(paired)))
    return list(map(float, np.quantile(paired[sample].mean(axis=1), [.025, .975])))


def study():
    assert sha256(Path(__file__).with_name("FAULT_PROTOCOL.md").read_bytes()).hexdigest() == PROTOCOL
    audit_fixture = rational_audit()
    source = banks()
    dev = evaluate(*source["development"])
    fixed = min(TRIPLETS, key=lambda triplet: np.mean([metrics[triplet]["worst_probe_width"]
                                                     for settings in dev for _, metrics in settings]))
    result = {}
    for name, (fields, seed, shift) in source.items():
        records = field_metrics(dev if name == "development" else evaluate(fields, seed, shift), fixed)
        means = {method: average([record[method] for record in records]) for method in records[0]}
        difference = means["median"]["worst_probe_width"] - means["minimax"]["worst_probe_width"]
        relative = difference / means["median"]["worst_probe_width"]
        interval = bootstrap(records)
        result[name] = {"fields": len(fields), "trajectories": 6 * len(fields), "means": means,
                        "worst_width_difference": difference, "relative_reduction": relative,
                        "paired_interval": interval,
                        "practical_gate_passed": bool(difference >= .005 and relative >= .10 and interval[0] > 0),
                        "field_paired_widths": [[a["median"]["worst_probe_width"], a["minimax"]["worst_probe_width"]]
                                                for a in records]}
    return {"protocol_sha256": PROTOCOL, "rational_audit": audit_fixture,
            "development_fixed": [MENU[i] for i in fixed], "cohorts": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(rational_audit(), indent=2))
        return
    result = json.dumps(study(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    elif result != args.verify.read_text():
        raise ValueError("Fault-tolerant telemetry replay differed from the saved results.")
    else:
        print("Fault-tolerant telemetry replay matches exactly.")


if __name__ == "__main__":
    main()

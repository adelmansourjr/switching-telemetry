"""Certify finite-window occupancy in an abstract two-relay switching system."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import tanh
from pathlib import Path
from random import Random

import numpy as np


PROTOCOL = "0f195975e5033616f7bb94eb9d8086e9ab6c565f76c816941d4a84831c338352"
TOL = 2e-9
MENU = ("mode_00", "mode_10", "mode_01", "first", "second", "xor")
GATES = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                  (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0)), dtype=float)
PUBLISHED = (((2, 1, 3, 3), (2, 2, 1, 4)),
             ((1, 2, 2, 1), (3, 2, 3, 4)),
             ((3, 2, 3, 2), (1, 1, 4, 4)))
NOMINAL = ("sign", "hysteresis", "smooth")
THRESHOLDS = np.array((.15, .25, .35, .45))
SIMPLEX = np.vstack((-np.eye(3), np.ones(3)))
SUBSETS = np.array([[int(bool(mask & (1 << i))) for i in range(3)]
                    for mask in range(1, 8)], dtype=float)
DIFFERENCE = np.vstack((SUBSETS, -SUBSETS))


class Vertices:
    """Cache the three active inequalities defining each candidate vertex."""

    def __init__(self, matrix):
        self.matrix = matrix
        active = np.asarray(list(combinations(range(len(matrix)), 3)))
        triples = matrix[active]
        independent = np.abs(np.linalg.det(triples)) > 1e-10
        self.active = active[independent]
        self.inverse = np.linalg.inv(triples[independent])

    def feasible(self, rhs):
        rhs = np.atleast_2d(rhs)
        vertices = np.einsum("vij,svj->svi", self.inverse, rhs[:, self.active])
        satisfy = vertices @ self.matrix.T <= rhs[:, None, :] + TOL
        return vertices, satisfy.all(axis=2)


class SensorModel:
    """Compute sharp conditional intervals and their maximum possible width."""

    def __init__(self, rows):
        self.rows = rows
        self.offset = rows[:, -1]
        reduced = rows[:, :3] - self.offset[:, None]
        directions = np.vstack((reduced, -reduced))
        self.conditional = Vertices(np.vstack((SIMPLEX, directions)))
        self.differences = Vertices(np.vstack((DIFFERENCE, directions)))

    def intervals(self, readings, radii):
        centered = np.atleast_2d(readings) - self.offset
        bounds = np.concatenate((centered + radii, -centered + radii), axis=1)
        base = np.tile([0., 0., 0., 1.], (len(centered), 1))
        vertices, feasible = self.conditional.feasible(np.hstack((base, bounds)))
        targets = 1. - vertices.sum(axis=2)
        lo = np.clip(np.where(feasible, targets, np.inf).min(axis=1), 0., 1.)
        hi = np.clip(np.where(feasible, targets, -np.inf).max(axis=1), 0., 1.)
        return np.column_stack((lo, hi)), feasible.any(axis=1)

    def certified_diameter(self, radii):
        budget = np.concatenate((np.ones(len(DIFFERENCE)), 2 * radii, 2 * radii))
        vertices, feasible = self.differences.feasible(budget)
        assert feasible.any()
        objectives = np.where(feasible[0], -vertices[0].sum(axis=1), -np.inf)
        best = int(np.argmax(objectives))
        difference = np.append(vertices[0, best], objectives[best])
        positive, negative = np.maximum(difference, 0.), np.maximum(-difference, 0.)
        remainder = (1. - positive.sum()) / 4.
        upper, lower = positive + remainder, negative + remainder
        assert min(upper.min(), lower.min()) >= -TOL
        assert abs(upper.sum() - 1) <= TOL and abs(lower.sum() - 1) <= TOL
        shared = self.rows @ ((upper + lower) / 2.)
        assert np.max(np.abs(self.rows @ upper - shared) - radii) <= TOL
        assert np.max(np.abs(self.rows @ lower - shared) - radii) <= TOL
        interval, valid = self.intervals(shared, radii)
        width = max(0., float(objectives[best]))
        assert valid[0] and abs(interval[0, 1] - interval[0, 0] - width) <= 8 * TOL
        return width


def drift_rows(field):
    a, b = field
    return np.array((np.asarray(a) * (1, -1, 1, -1),
                     np.asarray(b) * (1, 1, -1, -1)), dtype=float)


def trajectory(rates, initial, step, n_steps, width, kernel, lag):
    ax, ay = rates.tolist()
    x, y = initial
    u, v = int(x >= 0), int(y >= 0)
    history = [initial]
    totals = np.zeros(4)
    for t in range(n_steps):
        if kernel == "smooth":
            s, r = (1 + tanh(x / width)) / 2, (1 + tanh(y / width)) / 2
            p = ((1 - s) * (1 - r), s * (1 - r), (1 - s) * r, s * r)
        else:
            if kernel == "sign":
                u, v = int(x >= 0), int(y >= 0)
            elif kernel == "hysteresis":
                if abs(x) >= width:
                    u = int(x >= 0)
                if abs(y) >= width:
                    v = int(y >= 0)
            else:
                u = int(history[max(0, t - lag[0])][0] >= 0)
                v = int(history[max(0, t - lag[1])][1] >= 0)
            p = tuple(float(j == u + 2 * v) for j in range(4))
        totals += p
        x += step * sum(a * w for a, w in zip(ax, p))
        y += step * sum(a * w for a, w in zip(ay, p))
        history.append((x, y))
    occupancy = totals / n_steps
    displacement = (np.array((x, y)) - initial) / (step * n_steps)
    assert abs(occupancy.sum() - 1) <= TOL
    assert np.allclose(displacement, rates @ occupancy, atol=TOL, rtol=0)
    return occupancy, displacement


def generate_fields(seed, count, allowed, used):
    rng = Random(seed)
    fields = []
    while len(fields) < count:
        a, b = (tuple(rng.choice(allowed) for _ in range(4)) for _ in range(2))
        uncoupled = a[0] == a[2] and a[1] == a[3] and b[0] == b[1] and b[2] == b[3]
        if not uncoupled and (a, b) not in used:
            fields.append((a, b))
            used.add((a, b))
    return fields


def interval_scores(interval, valid, truth):
    lo, hi = interval.T
    covered = valid & (lo <= truth + TOL) & (truth <= hi + TOL)
    above = valid[:, None] & (lo[:, None] > THRESHOLDS + TOL)
    below = valid[:, None] & (hi[:, None] < THRESHOLDS - TOL)
    wrong = (above & (truth[:, None] <= THRESHOLDS)) | (below & (truth[:, None] >= THRESHOLDS))
    return {
        "width_sum": float(np.sum(hi[valid] - lo[valid])),
        "covered": int(covered.sum()),
        "empty": int((~valid).sum()),
        "threshold_decisions": int(np.sum(above | below)),
        "wrong": int(np.sum(wrong)),
    }


def field_results(field, rng, kernels, condition):
    rates = drift_rows(field)
    matrix = np.vstack((np.ones(4), rates))
    _, singular, right = np.linalg.svd(matrix)
    use_geometry = singular[-1] > 1e-9
    direction = right[-1]
    models = [SensorModel(np.vstack((rates, gate))) for gate in GATES]
    no_gate = SensorModel(rates)
    field_settings = []
    for _ in range(2):
        initial = np.array([rng.uniform(-.12, .12) for _ in range(2)])
        step = rng.choice((.04, .10))
        n_steps = rng.choice((24, 40, 64))
        width = step * rng.choice((.5, 1.5, 2.5))
        lag = (rng.randint(1, 3), rng.randint(1, 3))
        drift_radius = np.array([.018 * (2 ** rng.uniform(0., 3.)) for _ in range(2)])
        gate_radius = np.array([.004 * (2 ** rng.uniform(0., 4.)) for _ in MENU])
        if condition == "homogeneous":
            gate_radius[:] = .020
        if condition == "noisy_drift":
            drift_radius *= 3
        observed = [trajectory(rates, initial, step, n_steps, width, kernel, lag) for kernel in kernels]
        occupancy = np.array([sample[0] for sample in observed])
        displacement = np.array([sample[1] for sample in observed])
        noise = np.array([[rng.uniform(-1, 1) for _ in range(2 + len(MENU))]
                          for _ in kernels])
        common = displacement + noise[:, :2] * drift_radius
        telemetry = occupancy @ GATES.T + noise[:, 2:] * gate_radius * (4 if condition == "bias" else 1)
        truth = occupancy[:, 3]
        baseline, valid = no_gate.intervals(common, drift_radius)
        assert valid.all()
        candidates = []
        for index, model in enumerate(models):
            radius = np.append(drift_radius, gate_radius[index])
            diameter = model.certified_diameter(radius)
            readings = np.column_stack((common, telemetry[:, index]))
            interval, candidate_valid = model.intervals(readings, radius)
            assert np.all(interval[candidate_valid, 1] - interval[candidate_valid, 0] <= diameter + 8 * TOL)
            metrics = interval_scores(interval, candidate_valid, truth)
            if condition != "bias":
                assert metrics["covered"] == len(kernels) and metrics["wrong"] == 0
            candidates.append({"diameter": diameter, **metrics})
        geometry = np.abs(GATES @ direction) / gate_radius
        precision = int(np.argmin(gate_radius))
        choices = {
            "minimax": int(np.argmin(np.round([c["diameter"] for c in candidates], 11))),
            "null_projection": int(np.argmax(np.round(geometry, 11))) if use_geometry else precision,
            "precision_only": precision,
        }
        drift_metrics = interval_scores(baseline, valid, truth)
        assert drift_metrics["covered"] == len(kernels) and drift_metrics["wrong"] == 0
        field_settings.append({
            "trials": len(kernels),
            "choices": choices,
            "candidates": candidates,
            "drift_only": {"diameter": no_gate.certified_diameter(drift_radius), **drift_metrics},
        })
    return field_settings


def get_outcomes(setting, method, fixed):
    if method == "drift_only":
        return setting["drift_only"]
    if method == "uniform":
        # Integrate over the exact distribution of a uniformly drawn single channel.
        return {key: float(np.mean([c[key] for c in setting["candidates"]]))
                for key in setting["candidates"][0]}
    if method == "development_fixed":
        index = fixed
    elif method in MENU:
        index = MENU.index(method)
    else:
        index = setting["choices"][method]
    return setting["candidates"][index]


def summarise_field(field, settings, methods, fixed):
    trials = sum(s["trials"] for s in settings)
    rows = {}
    for method in methods:
        outcomes = [get_outcomes(s, method, fixed) for s in settings]
        counts = {key: sum(x[key] for x in outcomes)
                  for key in ("width_sum", "covered", "empty", "threshold_decisions", "wrong")}
        rows[method] = {
            "width": counts["width_sum"] / (trials - counts["empty"]) if counts["empty"] < trials else None,
            "diameter": float(np.mean([x["diameter"] for x in outcomes])),
            **counts,
        }
    return {"field": field, "trials": trials, "metrics": rows,
            "selectors": [{name: MENU[i] for name, i in s["choices"].items()} for s in settings]}


def analyse(field_bank, seed, kernels=NOMINAL, condition="nominal", fixed=None):
    rng = Random(seed)
    measurements = [field_results(field, rng, kernels, condition) for field in field_bank]
    if fixed is None:
        width_by_candidate = [sum(s["candidates"][j]["width_sum"] for group in measurements for s in group)
                              for j in range(len(MENU))]
        fixed = int(np.argmin(width_by_candidate))
    methods = ("minimax", "null_projection", "precision_only", "development_fixed",
               "uniform", "drift_only", *MENU)
    rows = [summarise_field(field, settings, methods, fixed)
            for field, settings in zip(field_bank, measurements)]
    total = sum(r["trials"] for r in rows)
    means = {}
    for method in methods:
        arm = [r["metrics"][method] for r in rows]
        counts = {key: sum(x[key] for x in arm)
                  for key in ("covered", "empty", "threshold_decisions", "wrong", "width_sum")}
        means[method] = {
            "mean_width_among_feasible": counts["width_sum"] / (total - counts["empty"]),
            "mean_exact_worst_case_width": float(np.mean([x["diameter"] for x in arm])),
            "coverage": counts["covered"] / total,
            "empty": counts["empty"],
            "wrong": counts["wrong"],
            "decision_fraction": counts["threshold_decisions"] / (len(THRESHOLDS) * total),
        }
    return {"fixed_gate": MENU[fixed], "trials": total, "fields": len(rows),
            "means": means, "per_field": rows}, fixed


def paired_intervals(report, challengers):
    fields = report["per_field"]
    samples = np.random.default_rng(120080).integers(0, len(fields), (4000, len(fields)))
    effects = {}
    for method in challengers:
        delta = np.array([row["metrics"][method]["width"] - row["metrics"]["minimax"]["width"]
                          for row in fields])
        boot = delta[samples].mean(axis=1)
        effects[method] = {
            "mean_reduction": float(delta.mean()),
            "relative_reduction": float(delta.mean() / report["means"][method]["mean_width_among_feasible"]),
            "paired_95_interval": np.quantile(boot, [.025, .975]).tolist(),
        }
    return effects


def exact_vertices(matrix, bounds):
    """Small rational oracle, eliminating no simplex variable."""
    for active in combinations(range(len(matrix)), 3):
        indices = (-1, *active)
        rows = [[Fraction(x) for x in (matrix[i] + [bounds[i]])] for i in indices]
        for i in range(4):
            pivot = next((j for j in range(i, 4) if rows[j][i]), None)
            if pivot is None:
                break
            rows[i], rows[pivot] = rows[pivot], rows[i]
            rows[i] = [x / rows[i][i] for x in rows[i]]
            for j in range(4):
                if i != j:
                    rows[j] = [x - rows[j][i] * y for x, y in zip(rows[j], rows[i])]
        else:
            solution = [r[-1] for r in rows]
            if all(sum(a * b for a, b in zip(row, solution)) <= b
                   for row, b in zip(matrix, bounds)):
                yield solution


def audit_fixture():
    rows = [[1, -1, 1, -1], [1, 1, -1, -1], [0, 1, 1, 0]]
    matrix = [[-int(i == j) for i in range(4)] for j in range(4)]
    bounds = [0] * 4
    centers, radii = [0, 0, Fraction(1, 2)], [0, 0, Fraction(1, 10)]
    for row, center, radius in zip(rows, centers, radii):
        matrix.extend([row, [-x for x in row]])
        bounds.extend([center + radius, -center + radius])
    vertices = list(exact_vertices(matrix + [[1, 1, 1, 1]], bounds + [1]))
    extrema = min(row[3] for row in vertices), max(row[3] for row in vertices)
    assert extrema == (Fraction(1, 5), Fraction(3, 10))
    solver = SensorModel(np.array(rows, dtype=float))
    interval, valid = solver.intervals(np.array(centers, dtype=float), np.array(radii, dtype=float))
    assert valid[0] and np.allclose(interval[0], np.array(extrema, dtype=float), atol=TOL, rtol=0)
    assert abs(solver.certified_diameter(np.array(radii, dtype=float)) - .1) <= TOL

    # States in units of half an Euler displacement, with rates of magnitude 1.
    def history(hysteretic_y):
        x, y, y_mode = 1, 1, 0
        modes = []
        for _ in range(6):
            x_mode = int(x >= 0)
            if not hysteretic_y or abs(y) >= 2:
                y_mode = int(y >= 0)
            modes.append(x_mode + 2 * y_mode)
            x += 2 * (1 - 2 * x_mode)
            y += 2 * (1 - 2 * y_mode)
        assert (x, y) == (1, 1)
        weights = [Fraction(modes.count(i), 6) for i in range(4)]
        assert weights[1] + weights[3] == weights[2] + weights[3] == Fraction(1, 2)
        return modes, weights

    sign, sign_weights = history(False)
    delayed, delayed_weights = history(True)
    assert sign_weights[3] - delayed_weights[3] == Fraction(1, 3)
    return {"rational_interval": [str(x) for x in extrema],
            "reachable_modes": [sign, delayed],
            "reachable_joint_duty": [str(sign_weights[3]), str(delayed_weights[3])]}


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return round(value, 10) if isinstance(value, float) else value


def experiment():
    fixture = audit_fixture()
    used = set(PUBLISHED)
    development = generate_fields(848322, 48, tuple(range(1, 8)), used)
    test = generate_fields(837612, 128, tuple(range(1, 8)), used)
    shifted_fields = generate_fields(882157, 64, (1, 2, 4, 8, 16), used)
    dev, fixed = analyse(development, 993812)
    nominal, _ = analyse(test, 993821, fixed=fixed)
    effects = paired_intervals(nominal, ("development_fixed", "null_projection", "precision_only", "drift_only"))
    primary = effects["development_fixed"]
    cohorts = {"development": dev, "nominal": nominal}
    for name, fields, seed, kernels, condition in (
        ("published", PUBLISHED, 993822, NOMINAL, "nominal"),
        ("held_out_delay", test, 993823, ("delay",), "nominal"),
        ("homogeneous_precision", test, 993824, NOMINAL, "homogeneous"),
        ("held_out_field_law", shifted_fields, 993825, NOMINAL, "nominal"),
        ("noisier_displacement", test, 993826, NOMINAL, "noisy_drift"),
        ("calibration_failure", test, 993827, NOMINAL, "bias"),
    ):
        cohorts[name], _ = analyse(fields, seed, kernels, condition, fixed=fixed)
    for name, cohort in cohorts.items():
        cohort["minimax_channel_counts"] = dict(Counter(
            setting["minimax"] for row in cohort["per_field"] for setting in row["selectors"]
        ))
        if name != "nominal":
            del cohort["per_field"]
    return rounded({
        "protocol_sha256": PROTOCOL,
        "fixture": fixture,
        "primary_passed": (nominal["means"]["minimax"]["coverage"] == 1.
                           and primary["mean_reduction"] >= .020
                           and primary["relative_reduction"] >= .15
                           and primary["paired_95_interval"][0] > 0.),
        "primary_paired_comparisons": effects,
        "cohorts": cohorts,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path)
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest() != PROTOCOL:
        raise ValueError("Protocol hash differs")
    if args.demo:
        fixture = audit_fixture()
        names = ("00", "10", "01", "11")
        print("Same initial and final state, with both switches individually active half the time.")
        for rule, modes, duty in zip(("Instantaneous", "Hysteretic"),
                                     fixture["reachable_modes"], fixture["reachable_joint_duty"]):
            print(f"{rule}: {' '.join(names[mode] for mode in modes)}  ->  joint duty {duty}")
        print("The joint duties give opposite answers to a threshold of 1/4.")
        return
    outcome = experiment()
    text = json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.verify:
        if args.verify.read_text() != text:
            raise ValueError("Replay differs from the saved outcome")
        print("Verified replay, exact width witnesses and reachable-history fixture.")
    else:
        args.write.write_text(text)
        print(json.dumps(outcome["primary_paired_comparisons"], indent=2))
        print("Primary gate passed:", outcome["primary_passed"])


if __name__ == "__main__":
    main()

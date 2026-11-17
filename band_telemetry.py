"""Certify nonmonotone decisions using every feasible support interval."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

import numpy as np

from fault_telemetry import interval_hull
from telemetry import GATES, PUBLISHED, SensorModel, drift_rows, exact_vertices, generate_fields
from two_fault_telemetry import (
    TOLERANCE, append_probes, bounded, choose_counters, honest_tape, new_banks,
)


PROTOCOL_SHA = "84da161aa148a256e0afc6bb63d00c2af178b86105a841ed9bddffda6f3722cb"
BANDS = np.array(((.10, .20), (.20, .30), (.30, .40), (.40, .50)))
SUPPORTS = tuple(s for length in (1, 3) for s in combinations(range(len(GATES)), length))


def band_decisions(components):
    lower, upper = components[..., 0, None], components[..., 1, None]
    start, stop = BANDS.T
    valid = lower <= upper
    contained = (lower >= start + TOLERANCE) & (upper <= stop - TOLERANCE)
    disjoint = (upper < start - TOLERANCE) | (lower > stop + TOLERANCE)
    present = valid.any(axis=0)
    inside = present & ((~valid) | contained).all(axis=0)
    outside = present & ((~valid) | disjoint).all(axis=0)
    return inside, outside


def disconnected(components):
    lower, upper = components[..., 0], components[..., 1]
    order = np.argsort(lower, axis=0)
    lower = np.take_along_axis(lower, order, axis=0)
    upper = np.take_along_axis(upper, order, axis=0)
    union_upper = np.maximum.accumulate(upper, axis=0)
    gaps = np.isfinite(lower[1:]) & (lower[1:] > union_upper[:-1] + TOLERANCE)
    return gaps.any(axis=0)


def exact_fixture():
    # The implicit final row in exact_vertices is the equality sum(p) = 1.
    reports = dict(zip((0, 1, 3, 4, 5), (1, 0, 0, 1, 0)))
    target_values = set()
    for support in combinations(reports, 3):
        rows = [[-int(i == j) for i in range(4)] for j in range(4)]
        bounds = [0] * 4
        for index in support:
            gate = [int(x) for x in GATES[index]]
            rows.extend((gate, [-x for x in gate]))
            bounds.extend((reports[index], -reports[index]))
        rows.append([1, 1, 1, 1])
        bounds.append(1)
        projection = {v[3] for v in exact_vertices(rows, bounds)}
        assert len(projection) <= 1
        target_values.update(projection)
    assert target_values == {0, 1}
    return {"possible_joint_occupancies": [0, 1], "hull": [0, 1],
            "band": [.4, .6], "union_decision": "outside", "hull_decision": "abstain"}


def evaluate_setting(rates, models, kernels, rng, widened):
    occupation, displacement, noise, drift_radius, gate_radius = honest_tape(rates, kernels, rng)
    if widened:
        drift_radius *= 3
    drift = displacement + noise[:, :2] * drift_radius
    gates = occupation @ GATES.T + noise[:, 2:] * gate_radius
    truth = occupation[:, 3]
    selected, _ = choose_counters(models, drift_radius, gate_radius)
    reports, _ = append_probes(models, selected, drift, gates, drift_radius, gate_radius, truth)
    components = np.stack([
        bounded(models[support], support, drift, reports, drift_radius, gate_radius)
        for support in combinations(selected, 3)
    ])
    covers_truth = ((components[..., 0] <= truth + TOLERANCE) &
                    (components[..., 1] >= truth - TOLERANCE)).any(axis=0)
    assert covers_truth.all()
    true_inside = (truth[:, None] >= BANDS[:, 0]) & (truth[:, None] <= BANDS[:, 1])
    union_inside, union_outside = band_decisions(components)
    hull_inside, hull_outside = band_decisions(interval_hull(components)[None])
    assert ((~hull_inside) | union_inside).all() and ((~hull_outside) | union_outside).all()
    assert not ((union_inside & ~true_inside) | (union_outside & true_inside)).any()
    additional = (union_inside | union_outside) & ~(hull_inside | hull_outside)
    return {
        "transcripts": int(covers_truth.size),
        "disconnected": int(disconnected(components).sum()),
        "band_queries": int(additional.size),
        "hull_certified": int((hull_inside | hull_outside).sum()),
        "union_certified": int((union_inside | union_outside).sum()),
        "additional_clean": int(additional[0].sum()),
        "additional_fault": int(additional[1:].sum()),
        "additional_by_band": additional.sum(axis=(0, 1)).tolist(),
    }


def cohort(fields, probe_seed, widened):
    rng = Random(probe_seed)
    kernels = ("delay" if widened else "sign", "hysteresis", "smooth")
    rows = []
    for field in fields:
        rates = drift_rows(field)
        models = {s: SensorModel(np.vstack((rates, GATES[list(s)]))) for s in SUPPORTS}
        first = evaluate_setting(rates, models, kernels, rng, widened)
        second = evaluate_setting(rates, models, kernels, rng, widened)
        rows.append({key: np.add(first[key], second[key]).tolist() for key in first})
    totals = {key: np.sum([row[key] for row in rows], axis=0).tolist() for key in rows[0]}
    differences = np.array([
        100 * (row["union_certified"] - row["hull_certified"]) / row["band_queries"]
        for row in rows
    ])
    draws = np.random.default_rng(742139).integers(len(rows), size=(10000, len(rows)))
    return {"fields": len(fields), "overall": totals, "wrong_calls": 0,
            "gain_pp": float(differences.mean()),
            "paired_field_interval_pp": np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist(),
            "per_field": rows}


def study():
    protocol = Path(__file__).with_name("BAND_PROTOCOL.md")
    assert sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA
    used = set(PUBLISHED)
    for seed, count, values in ((848322, 48, range(1, 8)), (837612, 128, range(1, 8)),
                                (882157, 64, (1, 2, 4, 8, 16))):
        generate_fields(seed, count, values, used)
    from fault_telemetry import banks
    for fields, _, _ in banks().values():
        used.update(fields)
    for fields, _ in new_banks().values():
        used.update(fields)
    nominal = generate_fields(890271, 48, range(1, 8), used)
    shifted = generate_fields(890281, 24, (1, 2, 4, 8, 16), used)
    fixture = exact_fixture()
    primary = cohort(nominal, 637081, False)
    return {"protocol_sha256": PROTOCOL_SHA, "fixture": fixture, "nominal": primary,
            "widened_calibration": cohort(shifted, 637111, True),
            "practical_gate_met": primary["gain_pp"] >= 1.0}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()
    data = study()
    result = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.verify:
        assert args.verify.read_text() == result
        print("Verified", args.verify)
    else:
        args.output.write_text(result)

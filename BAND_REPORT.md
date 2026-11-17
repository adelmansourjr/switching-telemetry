# Can a two-switch system be ruled out of an operating band?

An exact interval can hide gaps in the possible behaviour of two hidden switches. With five aggregate counters and up to two faulty readings, I constructed a transcript that permits only **0% or 100% joint occupancy**. The tightest interval spans the entire 0–100% range. Keeping the two possibilities separate certifies that joint occupancy is outside the 40–60% band. Rational enumeration verifies the fixture.

This extends the observability question in [Catsis, Hall and Jeffrey, *Perturbation and Determinacy of Nonsmooth Systems*](https://doi.org/10.1016/j.physd.2025.134679). Their drift equations constrain linear combinations of mode occupancies, while joint occupancy can depend on the switching implementation. Here the decision concerns membership in a fixed interval of joint activity.

## Exact statement

Each set of three counters defines a convex polytope when intersected with the occupancy simplex, the displacement bounds and the reported counter intervals. Its projection onto joint occupancy is a closed interval or empty. At most two of five counters can be faulty, so the full feasible set is precisely the **union** of the ten projections. A feasible occupancy satisfies at least one three-counter support, and every occupancy satisfying a support needs at most two arbitrary reports.

Taking the hull preserves extrema and therefore all one-sided threshold decisions. For a band, the union can certify exclusion when the band lies inside a gap. Both methods can certify inclusion under the same conditions. They use the same projections and measurement budget. This statement holds for the stated observation model, leaving physical reachability as an additional constraint.

For the fixture, the reports for mode-00, mode-10, first switch, second switch and XOR are respectively 1, 0, 0, 1 and 0. Use exact counters and uninformative displacement bounds. With XOR 0, modes 10 and 01 have zero mass, leaving only modes 00 and 11. A third agreement on this line requires one of its endpoints. Without the XOR agreement, any feasible triple among the four other counters has joint duty 0. Enumerating all ten triples in rational arithmetic gives exactly the two values above. This example establishes strictness under those observation constraints and has no physical validation.

## Fresh simulated holdout

The prospective comparison kept the probes, tolerances, error allowances, five counters and threshold bands fixed across the two decoders. Results include clean transcripts and engineered two-counter corruptions. [Frozen protocol](BAND_PROTOCOL.md) · [Complete results](band_results.json)

| Cohort | Transcripts | Disconnected projections | Correct band decisions, hull → union | Paired gain, percentage points | Wrong decisions |
|---|---:|---:|---:|---:|---:|
| Nominal calibration, 48 fields | 11,808 | 2,114 | 39,204 → 39,208 of 47,232 | 0.0085 [0.0000, 0.0233] | 0 |
| Widened calibration and delayed switching, 24 fields | 5,904 | 1,220 | 18,362 → 18,476 of 23,616 | 0.483 [0.119, 0.961] | 0 |

The nominal panel missed the prespecified 1-point gate. The disconnected projections were fairly common under these probes, while gaps rarely contained a full ten-point operating band. All added decisions were correct outside-band calls on engineered fault probes. The second panel changes both the displacement envelope and one switching implementation, so it cannot isolate the source of that panel's increase. Intervals resample whole fields and do not estimate how often these faults occur in hardware.

The exact fixture establishes a limit of interval-only reporting. The experiment quantifies how often that limit matters for four fixed bands under these simulated conditions. [Previous two-fault coverage and calibration checks](TWO_FAULT_REPORT.md)

## Replay

Python 3.12 and NumPy 2.5.4 generated the archived result. The replay generates the fields and uses no external data or model.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 band_telemetry.py --verify band_results.json
```

# Identifiability with faulty switching counters

The first [switching study](README.md) showed that an underestimated counter error can produce a narrow interval that excludes the truth. Here the displacement readings stay within their calibrated error bounds, and one of three additional counters can report any value. The aim is to certify the joint duty of two hidden switches under that fault model. On held-out simulated relay windows, the certified interval retained every truth and made zero wrong threshold calls across **7,488** clean and single-fault transcripts. Treating all three counters as calibrated made **76 wrong calls** and returned **6,025 empty sets**. The stronger comparison between two sound decoders gave only a **0.473 percentage-point** gain in mean worst-probe width, below the practical target. [Frozen protocol](FAULT_PROTOCOL.md) · [Complete results](fault_results.json).

The mathematical point builds on the drift and occupancy relationships of Catsis, Hall and Jeffrey. The redundancy argument connects to sparse-sensor error correction, where two rival states can have different suspect rows. The specific loss here is the sharp width of a **single occupancy target** on the simplex with bounded errors in the sound channels. [Nonsmooth occupancy paper](https://nonsmooth.land/wp-content/uploads/2025/05/2024_paradigm.pdf) · [Sparse-sensor error correction](https://doi.org/10.1109/TAC.2014.2303233).

The [two-fault follow-up](TWO_FAULT_REPORT.md) tests five counters under simultaneous faults and compares exact decoding with a safe pairwise approximation.

## The exact ambiguity bound

Let `p` and `q` be two four-mode occupancy vectors. Each chosen counter `i` has a matrix row `a_i` and error radius `ε_i`. The sound displacement rows always constrain both vectors. Among `m` extra counters permit at most `f` reports outside their advertised bounds. Then **the two vectors can share a transcript exactly when their displacement error boxes overlap and some `max(0, m − 2f)` extra counter error intervals overlap coordinatewise.**

To see necessity, a transcript consistent with both vectors must respect at least `m − f` counters for each. The two sets have at least `m − 2f` counters in common. On their overlap one reading must fit both predicted values. For sufficiency, choose a common set of that size. Put the corresponding readings at the midpoints between the two predictions. Partition the remaining counters into two groups of size at most `f`. On one group report the predictions of `p`, on the other those of (q). Each vector may disregard the group that reports the other prediction. Use midpoint readings for the always-sound displacement rows as well.

For fixed hardware, the worst-case target width is therefore the maximum of the original linear diameter program over these common counter subsets, with the displacement rows included in each program. For each subset the feasible differences have zero sum and ℓ₁ norm at most two. Apply the constraints `|Aδ| ≤ 2ε` only to the sound displacement rows and the selected common counters. Every such difference can be lifted to two occupancy vectors by taking its positive and negative parts and adding the same leftover mass to both. The subset construction above turns those vectors into a shared transcript. This proves sharpness in the relaxed simplex model. The program constructs a diameter-attaining pair and verifies the resulting conditional interval in every one of the **352** minimax calibration settings across development, nominal test and shift.

With **three counters and one fault**, the rule simplifies to the largest of three single-counter diameters. Minimising this score gives the value of the third-smallest single-counter diameter available in the menu. With **two counters and one fault**, there is a possible transcript for which neither counter reduces the worst-case displacement-only uncertainty.

A rational example makes the overlap visible. Set both sound marginal duties to 1/2. The occupancy vector is `(t, 1/2 − t, 1/2 − t, t)`. Use the mode-00, mode-10 and XOR counters, with radii `(1/8, 1/4, 1/16)`. The common readings `(0, 1/4, 0)` fit both `t = 0` and `t = 1/2`. At the first endpoint the XOR report is out of bounds. At the second it is the mode-00 report. The mode-10 reading is the only one in bounds for both. The rational demo checks these two hypotheses and exhausts **72** grid pairs over two calibrations, including pairs that cannot share a transcript. This fixture uses the observation simplex, while the earlier switching report separately supplies a pair of physically reachable relay histories.

## Comparing the decoders

At a given transcript the feasible set is a union of small polytopes, one per possible pair of sound counters. Reporting the smallest interval containing their projected union is sharp for the assumed constraints. A safe strong comparison is to take medians of the endpoint estimates from the three counters separately. Every possible truth lies in two of those single-counter intervals and hence in their median interval. The exact interval is nested inside that baseline for every valid transcript. The gain comes from enforcing that the same occupancy vector must satisfy both retained counters.

The primary evaluation used 96 previously unseen fields and 576 simulated trajectories, with sign, hysteretic and smooth kernels. The synthetic fault probes included a low reading, a high reading, a stuck midpoint, and a value selected from a displacement-compatible alternative. Every trajectory contributes 12 fault probes and one clean transcript. The interval below is computed by resampling entire fields. Methods use the same budget of three distinct extra readings.

| Method | Mean width across fault probes, pp | Mean worst-probe width, pp | Mean clean width, pp |
|---|---:|---:|---:|
| Exact pair consistency, minimax triplet | **3.236** | **5.290** | 3.923 |
| Safe median, same minimax triplet | 4.196 | 5.762 | 4.055 |
| Exact pair consistency, geometry triplet | 3.280 | 5.459 | 3.906 |
| Exact pair consistency, development-selected fixed triplet | 4.412 | 8.323 | 5.560 |
| Exact pair consistency, precision-selected triplet | 4.846 | 9.935 | 5.281 |
| Fault-location oracle, minimax triplet | 2.847 | 3.691 | 2.215 |

The primary gain was **0.473 points [0.388, 0.563]**, or **8.2%**, on the mean worst-probe width. The prespecified practical target was at least 0.5 points and 10%. Across all probes the exact method gave 567 more correct, resolved threshold calls than the median on the same transcripts. The exact and safe-median intervals each contained the true joint duty on every transcript. On the separate shift bank of 48 new fields with delayed switching and threefold displacement errors, the primary-style width fell from **9.564 to 8.781 points**. The gain over the geometry triplet stayed small. These are controlled simulation results on the stated four fault probes. The deterministic coverage guarantee applies to every report under the one-fault assumption and the calibrated displacement error boxes.

The clearest discussion question for Jeffrey is which additional physically reachable telemetry constraints can narrow the sharp simplex envelope when more than one counter can fail or calibration drifts together. The fault pattern and shared-displacement accuracy are distinct assumptions worth checking separately.

## Replay

The experiment imports the existing NumPy-only telemetry solver. It generates its own data. The reference Python and NumPy versions match the [original study](README.md).

```sh
python3 fault_telemetry.py --demo
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 fault_telemetry.py --verify fault_results.json
```

# Joint occupancy with two faulty counters

The previous [one-fault experiment](FAULT_TOLERANCE.md) left open what happens when two counters fail together. With five counters and calibrated displacement readings, a decoder that allowed two arbitrary counter faults retained the true joint duty on **15,744 of 15,744** fresh synthetic transcripts and made zero wrong threshold calls. A decoder that allowed only one fault on the same readings made **541 wrong threshold calls**. The main practical gain came from allowing the correct number of faults. An exact decoder and a strong, safe pairwise approximation differed by only **0.0077 percentage points** on the primary width metric, below the prespecified practical threshold. [Frozen protocol](TWO_FAULT_PROTOCOL.md) · [Results and field-level records](two_fault_results.json).

## What the five readings identify

The target is the proportion of a simulated window in which two hidden switches are active together. Both drift readings constrain the four nonnegative mode occupancies. At least three of the five extra counter readings must agree with any feasible occupancy. Project each feasible three-counter intersection onto the target and take the smallest interval containing all projected points. There is no tighter interval with the same coverage guarantee for this observation model.

Two feasible occupancies can explain a common transcript only if their displacement measurement boxes overlap and their three-counter support sets share at least one counter. Conversely, one shared counter with overlapping measurement bands is sufficient. Use their common midpoints for displacement and that counter, then assign the remaining two reports to the first occupancy and two to the second. This reduces the worst-case width of a five-counter design to the largest single-counter diameter. The code chooses the best design from the six available five-counter subsets before seeing any readings. This certificate uses the occupancy simplex. Not every occupancy in the simplex has to arise from a particular relay implementation.

The strongest comparison obtains the projected intervals for each pair in a candidate triple and intersects those one-dimensional intervals. It then pools over triples. This is safe and may be looser because pairs can have different explanations at the same projected target. For example, without informative displacement constraints, fix joint duty at 1/4 and impose mode-10 duty of 1/8, mode-01 duty of 1/8 and XOR duty of 3/8. Each pair has a feasible occupancy at that joint duty, while all three together would require XOR duty of 1/4. The experiment measures how much this distinction matters on generated observations.

## Held-out stress test

We used 64 previously unseen fields, two independent settings per field, and three relay implementations per setting. Each of the 384 simulated windows yielded a clean tape and 40 paired-fault probes, including four perturbations for every possible pair of faulty counters. All methods saw the same five channels and reports. The coherent probes set both faulty readings to agree with a single alternative occupancy. The exact interval contained both the actual and alternative joint duty on all **3,840** such probes. Their joint duties differed by a mean of **5.41 points** and a maximum of **39.26 points**.

| Method | Truths covered | Empty intervals | Correct calls | Wrong calls | Mean worst-probe width, pp |
|---|---:|---:|---:|---:|---:|
| Exact, two faults | 15,744 | 0 | 56,345 | 0 | 10.9182 |
| Pairwise envelope, two faults | 15,744 | 0 | 56,341 | 0 | 10.9259 |
| Median of five single-counter intervals | 15,744 | 0 | 54,854 | 0 | 11.4367 |
| Exact, one fault | 4,138 | 9,131 | 24,438 | 541 | — |

The one-fault method also produced **2,475 nonempty intervals that missed the truth**. Calls are counted across four fixed thresholds per transcript, so one transcript can contribute several calls. A call is made only if its whole interval falls strictly on one side of the threshold.

The exact solution narrowed the worst-probe interval by **0.0077 points** over the pairwise envelope, with a paired 95% whole-field bootstrap interval of **0.0008 to 0.0167 points**. That is a **0.070% relative** reduction. Both the 0.5-point and the 10% practical targets were missed. The coarser median baseline would have overstated the practical value of the more expensive exact geometry.

## Displacement calibration check

On 24 different fields with delayed sign, hysteretic and smooth trajectories, we tripled the displacement error while holding the five selected counters and reports fixed across decoders. With the actual widened radii the same exact decoder covered **5,904 of 5,904** transcripts and made zero wrong threshold calls. Leaving the advertised radii unchanged gave **3,764 of 5,904** coverage and **330 wrong calls**. Allowing two bad counters does not compensate for miscalibration in the displacement channels, which the decoder treats as sound. These constructed stress tests give no hardware failure rates. An instrumented bench could test both error envelopes using independently observed switching histories.

## Replay

The run uses Python 3.12 and NumPy 2.3.5, with no external data or model. The script checks support nesting, the design width bound, coherent alternatives, nominal coverage and deterministic replay against the archived JSON.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 two_fault_telemetry.py --verify two_fault_results.json
```

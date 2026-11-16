# Measuring joint occupancy in a system with two switches

**Start with the [latest two-fault stress test](TWO_FAULT_REPORT.md) and the [one-fault experiment and proof](FAULT_TOLERANCE.md).** Run the replays with Python 3.12 and the [pinned NumPy dependency](requirements.txt).

Two reachable switching histories can start and finish at the same state and give identical individual duty cycles while disagreeing on how often the two switches are active together. On a short, abstract two-relay system, I proved how to compute the exact worst-case uncertainty for the available readings and tested which single extra counter to install. Across 768 held-out simulated windows, the chosen counter reduced the mean interval for joint occupancy from **34.875 to 2.857 percentage points** relative to displacement alone. The stronger question was whether it paid to choose the counter per setting. The reduction over a fixed counter chosen on separate development fields was **0.707 points** (19.8% relative), below the prespecified 2-point practical threshold.

[Protocol fixed before evaluation](PROTOCOL.md) · [Saved results](results.json) · [Supervisor and paper notes](PAPER_NOTES.md) · [Follow-up on an unknown faulty counter](FAULT_TOLERANCE.md)

## Question and method

Catsis, Hall and Jeffrey show that even when two switches use independently specified rules, their simultaneous activation can depend strongly on the way the switches are implemented. Their drift equations do determine some weighted sums of the occupancies during sliding. The follow-up asks what we can certify about joint occupancy over a finite window when we can add a bounded-error aggregate measurement and leave the switching rule unspecified. It uses the paper's signed mode fields and the three coefficient sets from Figure 3 in a small descriptive panel. The main panel uses independent generated fields and short discrete windows. [Catsis, Hall and Jeffrey, 2025, sections 3.3 and 4](https://nonsmooth.land/wp-content/uploads/2025/05/2024_paradigm.pdf).

Each candidate sends one scalar giving the fraction of steps where its predicate holds. A predicate can be one specific mode other than `11`, either individual switch, or exactly one switch active (XOR). In the smooth system, it sums the corresponding continuous mode weights. Every selector receives the same two average displacement measurements and solves the same conditional interval problem. Before seeing any values, the proposed selector chooses the counter with smallest worst-case interval width over the advertised error boxes. The control based on the null space favours sensors most aligned with the displacement-invisible direction per unit of advertised error. The fixed control is the best single counter on 48 separate fields. The other controls choose uniformly, choose solely by precision, or use displacement alone. The budget is the number of transmitted scalars with comparable predicate hardware assumed.

## Exact result

Let $p$ be the four nonnegative mode fractions summing to one, and let $c$ select the fraction spent in `11`. Sum the Euler updates over the window to obtain $Vp=(x_T-x_0)/T$, where the columns of $V$ are the signed mode fields. Write all readout rows as $A$, including $V$ and any extra counter. For a received vector $z$, the feasible occupancies obey $|Ap-z|\leq\varepsilon$ componentwise. Minimising and maximising $c^\top p$ over this set gives the sharp interval under those constraints.

The **worst-case width over all compatible readings** is the value of a single linear optimisation:

```text
maximise     c · δ
subject to   sum(δ) = 0
             ||δ||₁ ≤ 2
             −2ε ≤ Aδ ≤ 2ε
```

Any two occupancies compatible with the same reading give a feasible difference `δ`. Conversely, for every feasible `δ`, take its positive and negative parts and add the same leftover probability mass to both. This yields two simplex vectors separated by `δ`. The midpoint of their predicted readings is within the error allowance of each, by the last inequality. The upper bound is therefore attained in the relaxed observation model. For each calibration and counter, the code constructs these two vectors and checks equality at their shared reading. For fixed advertised errors this minimises the worst-case interval **within the six available counters**. It requires no model of rule probabilities. Actual relay histories are a subset of the simplex model, so the optimum is for the stated relaxation. The small rational arithmetic check independently confirms a conditional interval of [1/5, 3/10]; the general vertex solver uses double precision with a feasibility tolerance of 2 × 10⁻⁹.

There is also an exact reachable example. Start both coordinates at half a step, with unit drift magnitude for each switch and no cross-coupling. With instantaneous relays the modes over six steps are `11 00 11 00 11 00`. Give the second relay hysteresis with thresholds one step either side of zero and an initially inactive memory, leaving the first instantaneous. The modes become `10 01 11 01 10 00`. Both systems return to the same start, each individual duty cycle is 1/2, and the `11` fractions are **1/2 versus 1/6**. Any interval based on those common summaries and valid for both histories needs width at least 1/3. Any single estimate errs by at least 1/6 on one history. A rule asking whether joint duty exceeds 1/4 receives opposite true answers on the two histories. An XOR counter distinguishes them. When the marginals are known, the joint duty is half their sum minus half the XOR duty.

## Held-out comparison

The primary panel has 128 independent generated fields. Every field contributes two settings with sign, hysteresis and tanh variants. Paired 95% intervals resample fields, keeping all their settings and kernels together. Fraction-of-window widths are displayed in percentage points. A threshold is declared only if the full interval lies strictly on one side of it.

| Selector | Observed mean width, pp | Mean worst-case width, pp | Thresholds resolved |
|---|---:|---:|---:|
| Minimum worst-case width | **2.857** | **2.946** | **94.60%** |
| Geometry divided by error radius | 3.061 | 3.182 | 94.04% |
| Fixed XOR selected on development | 3.563 | 3.699 | 93.00% |
| Smallest error radius | 6.000 | 7.165 | 88.15% |
| Uniform single counter, expected | 9.845 | 12.349 | 80.62% |
| Displacement only | 34.875 | 51.858 | 34.60% |

All nominal intervals contain the true joint occupancy. All acted-on threshold decisions are correct. At matched cost the absolute improvement over fixed XOR was **0.707 points [0.567, 0.857]**. It cleared the relative threshold and missed the 2-point absolute threshold. The improvement over the stronger geometry control was **0.205 points [0.115, 0.318]**. The main practical gain here was getting access to a joint aggregate. The difference between the two strongest selection policies is small for this sensor menu.

| Secondary setting | Proposed width, pp | Fixed XOR width, pp | Geometry control width, pp |
|---|---:|---:|---:|
| Published coefficient sets, 18 simulated windows | 2.380 | 4.268 | 2.997 |
| Held-out delayed sign rule, 256 windows | 3.057 | 3.758 | 3.307 |
| Held-out field law, 384 windows | 2.084 | 2.784 | 2.214 |
| Common displacement radii tripled, 768 windows | 6.077 | 6.890 | 7.000 |
| Same precision for all six counters, 768 windows | 3.509 | 3.527 | 3.519 |

All these panels preserved 100% coverage for the proposed selector. When every counter had the same precision, the adaptive rule chose XOR in 238 of 256 settings, and its gain over fixed XOR shrank to **0.018 points**. In the error misspecification test, the realised counter error was drawn at up to four times its advertised bound. Coverage fell to **61.59%**, 14 intervals were empty, and 39 threshold decisions were wrong. Therefore an interval can be narrow and nonempty while giving a wrong answer when the error bound is too small. A next empirical test would calibrate the aggregate counters separately on an instrumented bench and check coverage on a new set of switching schedules. [Complete cohort outcomes](results.json).

## Replay

Use Python 3.10 or later with NumPy. The saved run used NumPy 2.3.5. The demo runs the exact witness and rational fixture. The full replay generates the field split, tests the error assumptions, verifies difference witnesses and compares results with the archived JSON. It uses generated data and requires only the indicated Python dependency.

```sh
python3 telemetry.py --demo
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 telemetry.py --verify results.json
```

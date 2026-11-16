# One telemetry channel for hidden switching

## Scope and question

Catsis, Hall and Jeffrey derive occupancy combinations that are comparatively stable under regularisation of a four-mode switching system. The earlier repository pilot bounds joint occupancy from drift balance. Here the question is whether choosing one extra bounded-error scalar occupancy summary, using only the known mode fields and sensor error bounds before the window, reduces the uncertainty in simultaneous activation. The new contribution is an exact worst-case diameter over the deliberately relaxed simplex of compatible mode fractions. The generated coupled cohort is the empirical test of whether this worst-case criterion also narrows typical realised intervals.

Each method receives two common mean-displacement readings and, except for the zero-extra control, one additional scalar per window. Each extra channel has a fixed interpretation and advertised error bound for the trial. The counter integrates a binary mode predicate over the window. For the smooth kernel it integrates the same predicate applied to continuous mode weights. This is a matched number of transmitted scalars with equivalent predicate hardware assumed. The primary metric measures precision in that observation model.

## Data and observations

Order modes 00, 10, 01, 11. The estimand is the fraction of weight on 11 in a short Euler window. For rates a and b the signed rows are (a0, −a1, a2, −a3) and (b0, b1, −b2, −b3). Use the three published rate sets from Catsis et al., Figure 3, as a separate case bank. The generated coupled field distribution draws each rate independently as a uniform integer from 1 through 7. Reject exactly uncoupled fields, any duplicate, and any field in the published bank. Draw 48 development fields with Python random seed 848322 and 128 new test fields with seed 837612. The two data sets are field-disjoint.

For each field simulate two independent settings. Draw each initial state coordinate uniformly in [−0.12, 0.12], a time step from {.04, .10}, an Euler step count from {24, 40, 64}, and a hysteresis/smoothing width from {.5, 1.5, 2.5} step units. Simulate sign, hysteretic and smooth tanh kernels, each with the same paired settings and independent measurement errors. The calibration and observation streams use seeds 993812 for development, 993821 for test, 993822 for the published bank, 993823 for the held-out kernel, 993824 for homogeneous error, 993825 for held-out field law, 993826 for the noisier common-channel stress, and 993827 for the bias stress. The generator has a single explicit order per seed. Use an independent paired bootstrap with 4,000 field-level resamples and NumPy seed 120080.

The mean-displacement readings are V times occupancy plus independent uniform errors. Their nominal radii are independently .018 times 2 raised to a uniform power in [0,3]. Candidate telemetry rows are mode 00, mode 10, mode 01, first switch active, second switch active, and exactly one switch active. They each report their row applied to the common occupancy plus independent uniform error. Each independently known radius is .004 times 2 raised to a uniform power in [0,4]. Sensors are chosen before any displacements or occupancy values are revealed. Pair comparisons on the same field, simulation, noise realisations and thresholds. The generator preassigns errors to every candidate solely to permit paired counterfactual comparisons.

## Selectors and analysis

Each method uses the same sharp interval over the simplex consistent with its available readings and advertised error bounds.

1. Choose the candidate with the smallest exact worst-case feasible interval diameter given V and the error radii. Ties use menu order. Obtain the diameter by solving for a feasible difference of two occupancies; reconstruct one common transcript as a tightness audit.
2. Strong geometry control. Choose largest absolute projection of the telemetry row onto the one-dimensional null space of the unit-sum and two drift equations, divided by the candidate error radius. Use precision-only selection if rank leaves a larger null space. This control uses identical calibration information.
3. Precision control. Choose the channel with smallest advertised error radius.
4. Fixed control. For each candidate compute average realised width in development. Freeze the best global fixed channel by that metric and evaluate it on test. Also report all six fixed candidate widths, and the expected width when uniformly choosing one channel.
5. Drift-only control uses the two common measurements. No extra channel.

Primary endpoint is the per-field mean difference in interval width between proposed choice and development-selected fixed control on the 128 new coupled fields. Require 100% coverage in the nominal cohort for the proposed intervals, at least .020 absolute and 15% relative narrower mean width than fixed control, and positive lower endpoint of the paired 95% field bootstrap interval. The geometry control is reported prominently and its paired interval is diagnostic. Comparison to all fixed test choices is descriptive. The model and primary criterion stay fixed across reruns.

Also report the exact worst-case certified widths, strict threshold decisions with the common fixed thresholds {.15,.25,.35,.45}, and wrong decisions on every condition. A method acts only if its interval lies entirely on one side of the threshold. No learned switching-kernel classifier is involved.

## Checks and tests of the assumptions

- Independently solve the exact two-history witness in the report. Same initial state, displacement, and two marginal duty fractions can conceal a material change in joint occupancy.
- Test a delayed relay with 1–3 Euler steps per switch, unknown to the selection rule. Test the three published fields separately. Test uniform extra-channel error .020 to show the dependence on sensor heterogeneity, and 3× mean-displacement error bounds to stress the geometry control. Test 64 additional fields from an independent random stream 882157 with positive rates drawn from {1, 2, 4, 8, 16}, excluding previously used fields. These are descriptive.
- In a calibration failure stress, the nominal telemetry error bound is deliberately exceeded. Multiply an independent uniform signed telemetry error within the bound by 4 while fitting with the original radius. Record excluded truths and intervals that are empty. An empty interval produces no action.
- For every feasible primary case, audit space containment, the drift identity, and that realised width never exceeds the exact diameter for that configuration. For every selection configuration, construct difference witnesses and check that shared-transcript intervals attain the computed diameters. Use a separate rational exhaustive vertex oracle on a small integer fixture. Document double-precision tolerances and report ambiguity at thresholds conservatively.

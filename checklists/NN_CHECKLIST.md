# Neural Network and Epsilon Predictor Checklist

This is a separate implementation track from [P0 — Production Core](P0_CHECKLIST.md).
P0.8 and P0.9 accept **externally supplied predicted epsilon**: they implement
mathematically correct reverse steps, not a trained model. An actual learned
epsilon predictor is required before end-to-end image generation.

Aligned with [ROADMAP.md](../ROADMAP.md), especially Milestones 3, 4, 5, 6 and 7. Image encoding, decoding, reconstruction training and latent handoff are separately tracked in [AUTOENCODER_CHECKLIST.md](AUTOENCODER_CHECKLIST.md).

## NN.1 — Trainable neural operations (Roadmap Milestone 3)

- [x] Define a production `Layer` interface with shape contracts.
- [x] Implement FP64 Conv2D forward and backward propagation (NN.1.04–05).
- [x] Implement FP64 ReLU, SiLU and Linear with validated backward passes (NN.1.02–03C).
- [ ] Implement spatial upsampling/transposed convolution as required.
- [x] Define parameter, gradient and zero-grad interfaces.
- [x] Integrate production AdamFP64 with model Parameters (NN.1.06); deterministic initialisation is separately pending in NN.1.07B.
- [ ] Implement policy-governed, opt-in gradient geometry controls (NN.1.07C/D); existing layer/optimiser finite-value guards are validated.
- [x] Test validated NN.1.01–06 layer shapes, invalid inputs and immutability where applicable.
- [x] Validate Linear, ReLU/SiLU and Conv2D gradients against finite differences.
- [ ] Demonstrate deterministic loss reduction as an NN.1.07F smoke test under the reviewed training policy.
- [ ] Add Doxygen public API documentation.

**Exit criterion:** A small CPU network trains reproducibly and passes numerical
gradient and optimiser tests.

### NN.1.03–NN.1.07 — Small-ticket implementation plan

#### NN.1.03A — ReLU forward and backward (signed off 20 November 2026)
- [x] Verify ReLU known-answer forward and backward for negative, zero, positive inputs.
- [x] Verify Tensor shape preservation, no parameters and input immutability.
- [x] Verify invalid-shape, missing-forward, NaN/Inf rejection.
- [x] Run `bash scripts/test.sh --nn`, `--p0`, and `--all`.
- [ ] Verify Doxygen generation separately (not reported in this run).

**Validation:** User reports `bash scripts/test.sh --nn`, `--p0`, and `--all` passed after the ReLU integration. Expected registered test counts: 3 NN / 7 P0 / 11 total CPU; exact test output and timings not supplied. Doxygen generation was not separately confirmed.

#### NN.1.03B — SiLU (Swish) (CPU Debug signed off 20 November 2026)
- [x] Validate stable FP64 SiLU forward and backward against known answers.
- [x] Validate output and gradient shapes and preservation of input.
- [x] Validate extreme-value and invalid-input behaviour.
- [x] Run independent `nn_silu` plus `--nn`, `--p0`, and `--all` suites.
- [ ] Confirm optional Doxygen generation separately.

**Validation:** User reports all NN, P0 and full CPU regression suites passed after integration of `nn_silu`. The expected registered counts are 4 NN, 7 P0 and 12 total CPU; exact logs/timings were not supplied. Doxygen generation, Release, Metal and clean-clone validation remain unverified.

#### NN.1.03C — Activation numerical validation (CPU Debug validated 20 November 2026)
- [x] Validate independent central finite differences for ReLU away from zero and SiLU across negative, zero and positive inputs.
- [x] Verify ReLU's documented zero-subgradient convention.
- [x] Validate SiLU extreme-value gradients and rank-4 shape preservation.
- [x] Confirm failed forward calls do not corrupt the cached valid input.
- [x] Run `bash scripts/test.sh --nn`, `--p0`, and `--all`.

**Validation:** User confirms full CPU suite PASS (13/13), including the `nn_activation_validation` test registered under the `nn` label. Numerical-gradient, edge-case, rank-4 and forward-cache validation cases passed as part of that suite. Individual `--nn` / `--p0` counts and timings not supplied for this run; Doxygen, Release and Metal remain separately unverified.

#### NN.1.04 — Conv2D forward (CPU Debug signed off 20 November 2026)
- [x] Define NCHW convolution shape, kernel, stride and padding contracts.
- [x] Implement FP64 Conv2D forward and independent `nn_conv2d_forward` test.
- [x] Validate multi-channel, multi-output, batch, stride, padding and known-answer convolution results.
- [x] Correct the header filename from `Cov2D.hpp` to `Conv2D.hpp`.
- [x] Run `bash scripts/test.sh --nn`, `--p0` and `--all` after the header fix.

**Validation:** User confirms all three CPU test suites passed after correcting the header filename. Expected registered counts were 6 NN, 7 P0 and 14 total CPU; exact output/timings were not supplied. Doxygen generation, Release and Metal remain separately unverified. Backpropagation is tracked in NN.1.05.

#### NN.1.05 — Conv2D backward (CPU tests passed, 20 November 2026)
- [x] Implement gradients for input, kernel and bias.
- [x] Add known-answer and finite-difference `nn_conv2d_backward` test.
- [x] Verify batched gradient accumulation and `zero_grad()`.
- [x] Verify invalid backward inputs are rejected without corrupting stored gradients.
- [x] Confirm NN, P0 and full CPU regressions pass.

**Validation:** User reports all test suites passed after the NN.1.05 implementation. Verified `Conv2D.hpp`, `Conv2D.cpp`, the new test and CMake/CTest registration on GitHub. Registered test counts expected: 7 NN, 7 P0 and 15 full CPU; exact output/timings not supplied. Release, Metal, Doxygen and clean-clone validation not separately confirmed.

#### NN.1.06 — FP64 Adam (CPU signed off 20 November 2026)
- [x] Add production `AdamFP64` for FP64 `Parameter` tensors while retaining experimental FP32 Adam separately.
- [x] Validate first-step known answers, multiple steps, deterministic replay and multiple parameter tensors.
- [x] Validate optimiser reset, `zero_grad()`, invalid hyperparameters and non-finite gradients.
- [x] Validate all-or-nothing failed updates and subsequent recovery, and integration with `Linear`.
- [x] Record full CPU regression PASS (16/16).

**Validation:** User reports all 16 registered CPU tests passed after integrating NN.1.06. Verified the production header, source, independent test and CMake registration in GitHub. Individual test logs, run times, Doxygen generation, Release, Metal and clean-clone validation were not supplied.

### NN.1.07 — Policy-governed training and geometric optimisation (design planned, not implemented)

**Design principle:** Define the training policy first, then separately implement initialisation, gradient geometry, update constraints, lifecycle and empirical validation. A toy loss-reduction test is an engineering diagnostic, **not** the training policy or evidence of a trained image generator. No clipping mode is enabled implicitly.

**Architecture reference:** [Training Architecture — NN.1.07A Design v0.1](../docs/TRAINING_ARCHITECTURE.md). The architecture is documented; all implementation and validation tasks remain open.

#### NN.1.07A — Training policy specification and schema (first)
- [x] Document proposed training system architecture, model boundaries, geometric metric and lifecycle (design only; [architecture](../docs/TRAINING_ARCHITECTURE.md)).
- [x] Implement versioned `TrainingPolicy` schema and validation rules (NN.1.07A; CPU regression passed 17/17).
- [ ] Specify objectives for autoencoder reconstruction versus diffusion epsilon prediction, with distinct training and validation metrics.
- [ ] Specify dataset manifests, provenance, train/validation/test isolation, preprocessing, deterministic shuffling and random seeds.
- [ ] Specify optimisation policy: FP64 precision, AdamFP64 hyperparameters, schedules, accumulation steps, invalid-step rejection and stopping limits.
- [ ] Specify explicit independent modes for gradient constraints (`none`, `euclidean`, `metric_radial`) and optimiser-update constraints (`none`, `trust_region`); default to disabled until reviewed.
- [ ] Specify run logging, checkpoint contents, resume guarantees, failure recovery and promotion/acceptance criteria; record code revision and policy version.
- [x] Confirm user-reported 17/17 full CPU regression including `nn_training_policy` parser/serialization/invalid-config tests (20 November 2026).
- [ ] Confirm optional Doxygen generation separately (not included in regression evidence).
- [ ] Review and approve policy **before** introducing training execution or geometric clipping.

**Implementation evidence (CPU regression validated; full policy scope remains open):** `latent/nn/training/TrainingPolicy.hpp`, `src/nn/training/TrainingPolicy.cpp`, `tests/nn/test_training_policy.cpp`, and CMake registration added on 20 November 2026. Configuration only: no trainer, geometric clipping or checkpoint I/O has been implemented. User reports full CPU regression PASS 17/17; individual NN/P0 suite counts and Doxygen generation not separately supplied. Broader policy features still unchecked above require separate design/implementation, including update-constraint selection and complete runtime/checkpoint requirements.

**Exit:** A reviewed, testable training-policy contract independent of any particular dataset or model architecture.

#### NN.1.07B — Deterministic parameter initialisation (CPU validated 20 November 2026)
- [x] Implement seeded FP64 He/Xavier initialisers for Linear and Conv2D, with fan-in/fan-out scaling and zero biases.
- [x] Verify seed/layer-ID reproducibility, order independence, finite nonzero weights, statistics, gradient reset and invalid-input safety in the registered `nn_initialisation` test.
- [x] Record initialisation metadata (master/derived seed, layer identifier, method, fan-in/out, standard deviation) in `InitialisationRecord`.
- [ ] Wire initialisation metadata and RNG/checkpoint state into the future training lifecycle (NN.1.07E).
- [ ] Remove duplicate `Initialisation.cpp` compilation from the `test_initialisation` executable CMake target (already included in `latent_core`); review before modification.
- [ ] Verify optional Doxygen generation, Release, Metal and clean-clone separately.

**Validation:** User reports full CPU regression PASS (18/18, 20 November 2026). Header `latent/nn/training/Initialisation.hpp`, implementation `src/nn/training/Initialisation.cpp`, test `tests/nn/test_initialisation.cpp`, and CMake registration were verified present on the GitHub default branch. NN.1.07B core implementation is CPU validated; checkpoint integration and the CMake cleanup remain open. The full suite was user-run; no independent CI evidence was supplied.

#### NN.1.07C — Gradient geometry interface and Euclidean reference (CPU validated 21 November 2026)
- [x] Implement read-only FP64 global L2 gradient measurement for registered `Parameter` tensors in `GradientController`.
- [x] Implement opt-in global-norm radial clipping with zero-norm, non-finite, invalid registration and numerical overflow safeguards.
- [x] Add `nn_gradient_controller` known-answer, multi-parameter, threshold, read-only measurement, failure preservation and large-value tests; user confirms full CPU regression PASS (19/19).
- [x] Expose per-application diagnostics including pre/post norm, scale, clipped flag, parameter count and block count.
- [ ] Integrate session-level clipping frequency and rejected-step logging with the future training lifecycle (NN.1.07E).
- [ ] Review/refine post-clipping measured norm and stage/commit exception guarantees before treating the controller as production-atomic beyond the covered tests.
- [ ] Verify Doxygen generation, Release and Metal separately.

**Validation:** User reports clean-build full CPU CTest PASS (19/19, 21 November 2026) after hierarchical CMake refactor. `GradientController` and `nn_gradient_controller` are included in `latent_core` and the NN CTest label respectively. This establishes the Euclidean reference baseline for NN.1.07D, but not validation of the future metric-radial controller.

#### NN.1.07D — Diagonal and block-metric gradient controller (CPU validated, 21 November 2026)

**Evidence:** Earlier NN 12/12 PASS predates final numerical corrections. After corrections user ran Debug `--nn` (11/12 PASS) and `--all` (19/20 PASS) on 21 November 2026; only `nn_metric_gradient_controller` failed with `Expected rejection missing`. Root cause: the extreme-value test incorrectly expected an overflow for a single block RMS of `[DBL_MAX, DBL_MAX]`, which is mathematically finite. Corrected to assert successful finite clipping; also silenced an intentional `[[nodiscard]]` rejection test. The corrected test has now passed as part of the user-confirmed full CPU suite (20/20).

- [x] Define fixed positive-definite equal-block-RMS diagonal metric `M_b = I/(B*d_b)`, stable parameter ownership/shape registration and metric version `equal_block_rms_v1`.
- [x] Implement opt-in metric-norm radial scaling using `sqrt(g^T M g)` with scaled sum-of-squares accumulation and overflow checks.
- [x] Measure actual staged post-clipping metric norm before the nonthrowing gradient commit; reject weighted-zero underflow, clipping-scale underflow, clipping-to-zero underflow and non-finite values without changing registered gradients.
- [x] Extend independent NN test to cover measured post-norm, read-only measurement, no-op, validation failures, underflow, overflow, large finite inputs and atomic failure preservation.
- [x] Document that metric-radial scaling is **not generally the Euclidean closest-point projection onto an ellipsoid**; Fisher/natural-gradient approximations remain out of scope.
- [x] Re-run full CPU CTest after correcting the extreme-value fixture: **20/20 PASS** reported by user on 21 November 2026, including all 12 NN tests.
- [ ] Perform fixed-seed Euclidean-vs-metric training comparison as part of NN.1.07F; baseline remains independent and unchanged.
- [ ] Doxygen, Release and Metal validation remain separately unverified.

#### NN.1.07E — Image-first training execution lifecycle and checkpoints

**Design:** [IMAGE_TRAINING_LIFECYCLE_NN_1_07E.md](../docs/IMAGE_TRAINING_LIFECYCLE_NN_1_07E.md) (21 November 2026). Target: C++17 CPU FP64 reference, **images first**. E.01 deterministic image data source is now implemented and registered; no executable trainer, checkpoint codec, real image AE or denoiser is implemented yet. Video uses a separate future temporal track.

##### NN.1.07E.01 — Image tensor, source and provenance (CPU validated, 21 November 2026)
- [x] Implement `ImageBatch` as rank-four NCHW FP64 Tensor with ID-count match, finite [0,1] pixels and positive dimensions inherited from `Tensor`; verify in registered test.
- [x] Implement in-memory `ImageDataset` with uniform 1/3-channel CHW samples, globally unique IDs, mandatory nonempty train/validation/test splits and deterministic content fingerprint. Current fingerprint is FNV-1a (mismatch detection, **not** authenticated provenance or a saved file manifest).
- [x] Implement deterministic `ImageBatchSource` with fixed SplitMix64 Fisher–Yates per train epoch, held-out canonical ordering, partial final batches, validated versioned cursor restore and no cursor advancement before batch assembly succeeds.
- [x] Add synthetic known-pixel fixture and registered `nn_image_dataset` CTest covering sample mapping, seeded replay, restored cursor, split isolation, malformed cursors and invalid batches/datasets.
- [x] **Validation gate:** User confirms full CPU regression **21/21 PASS** on 21 November 2026, including registered `nn_image_dataset` and the 13 NN-labelled tests. Individual suite output and Release/Metal/Doxygen were not separately supplied.
- [ ] **Future real-data work:** image file decoding, cryptographic content manifests, on-disk split verification and optional augmentation are outside E.01's in-memory reference scope.

##### NN.1.07E.02 — Task interface and toy image-reconstruction fixture (CPU validated, 21 November 2026)
- [x] Add Doxygen-documented `ImageTrainingTask` interface: task ID, stable registered Parameters, unscaled microbatch mean loss with explicit gradient scaling, read-only held-out validation and cache-reset hook.
- [x] Add `ToyImageReconstructionTask`: flatten validated NCHW into [N, C*H*W], train an existing FP64 `Linear` layer with image-element mean-squared error, accumulate scaled analytical gradients and report unscaled loss. This is a linear reconstruction **fixture**, not an autoencoder or generative image model.
- [x] Provide non-mutating held-out validation that reads parameter values through newly added const `Linear` accessors without changing training forward-cache or gradients.
- [x] Add guarded microbatch exception path: snapshot gradients before backward, restore both weight and bias gradients when a failure occurs. This is not yet a whole-window / optimiser transaction.
- [x] Add independent `nn_image_reconstruction_task` CTest: one-pixel analytical weight/bias known answer, multi-image/multi-feature loss/gradient, accumulating scales, deterministic replay, validation immutability and invalid config/failure atomicity.
- [x] **Validation gate:** User-supplied Debug test log confirms `bash scripts/test.sh --nn` **14/14 PASS** and `bash scripts/test.sh --all` **22/22 PASS** (21 November 2026). `nn_image_reconstruction_task` passed in both runs; no build warnings reported.
- [ ] Actual image autoencoder architecture, learned encoder/decoder and visible reconstructions remain in AE track.

##### NN.1.07E.03 — Sample-weighted gradient accumulation (CPU validated, 21 November 2026)
- [x] Implement `GradientAccumulator`: calls `train_microbatch(batch,n_j)` and normalises the resulting gradient tensors once by `N=sum_j n_j`. Uses the actual final partial-batch sample count and reports a sample-weighted mean loss.
- [x] Reject nonzero incoming gradients, unsupported/empty parameter registration, duplicate/null parameters, non-finite values, sample count overflow and non-representable FP64 counts before committing a window.
- [x] Snapshot entire deterministic `ImageBatchSource` and registered parameter values/gradients, then restore using noexcept move operations if a microbatch, model, loss or normalisation fails. `reset_transient_state()` called after failure.
- [x] Leave optimiser stepping, global clipping, epoch advancement and full training-session transactions for E.04–E.07; a successful E.03 window produces only normalised gradients and diagnostics.
- [x] Add `nn_gradient_accumulator` CTest: eight 16-image microbatches (128 images), 2+2+1 sample-weighted partial window, short last batch with full microbatch count, zero-window end-of-epoch, seeded replay, nonzero initial gradients, and deliberate mid-window failures (exception, invalid loss, non-finite gradient, model mutation) with cursor/parameter rollback.
- [x] **Validation gate:** User confirms full CPU regression **23/23 PASS** (21 November 2026), including registered `nn_gradient_accumulator` and 15 NN-labelled tests. The exact per-test log and separate Release/Metal/Doxygen validation were not supplied.
- [ ] Broader task-owned RNG, external side effects and cross-component Adam transaction are deferred to E.07; E.03 rollback covers registered parameters and deterministic in-memory batch source only.

##### NN.1.07E.04 — TrainingPolicy/runtime execution gate (CPU validated, 21 November 2026)
- [x] Preserve policy-v1 `TrainingPolicy::validate()`, `serialize()` and `parse()` disabled-only contract and canonical field format. Enabled requests are validated by the new separate `TrainingExecutionGate`; the gate performs **no** execution or optimiser operation.
- [x] Require `execution_enabled=true`, otherwise reject; validate every pre-existing numeric, enum, policy metadata and metric constraint through a disabled clone of the same policy before authorising.
- [x] Match requested manifest fingerprint to actual `ImageDataset`, require canonical `train`/`validation`/`test` roles, validate supported `ReconstructionMSE` objective, matching `ToyImageReconstructionTask` class/task ID and `[C,H,W]` shape, and ensure non-null distinct finite registered parameters.
- [x] Add Doxygen contracts and `nn_training_execution_gate` CTest covering enabled/disabled policy, unchanged policy-v1 roundtrip, supported gradient modes, dataset/split/task/shape mismatches, invalid hyperparameters and non-finite parameters.
- [x] **Validation gate:** User confirms full CPU regression **24/24 PASS**, including the new `nn_training_execution_gate` (16 NN CTest registrations). Separate Release, Metal and Doxygen verification remain unconfirmed.
- [ ] **Follow-up:** Executable policy checkpoint format/version, arbitrary image model support and full training-session execution belong to E.07–E.08. This E.04 gate is an immutable compatibility record, **not** a claim that a full runtime is available.

##### NN.1.07E.05 — Gradient control and metrics (CPU validated, 21 November 2026)
- [x] Add Doxygen-documented `TrainingGradientControl` with stable, distinct non-owning parameter registration and mode selection: `None`, `EuclideanGlobal`, and `MetricRadial` with `equal_block_rms_v1`.
- [x] Reuse existing NN.1.07C/D implementations unchanged. A unified `TrainingGradientMetrics` records mode, norm kind, versioned metric ID, pre/post norms, applied scale, clipping flag, parameter-block count and scalar-element count.
- [x] Measure **actual final** post-clipping norm by rereading gradients after application, including Euclidean mode where the underlying NN.1.07C `GradientReport::norm_after` currently reports the target threshold. The `None` mode reports unchanged Euclidean L2 norm.
- [x] Count successful controller applications, clipped applications and rejected applications separately; these are not optimiser commits.
- [x] Stage previous gradients and restore on control/post-measure failure, with noexcept Tensor move verification. Stable registration validates shapes and finite model/gradient values.
- [x] Add independent `nn_training_gradient_control` CTest for the three modes, mathematical fixtures, actual post-norm, controller metadata, invalid configurations, non-finite gradients, destructive underflow rejection and counter behaviour.
- [x] **Validation gate:** User confirms full CPU suite **25/25 PASS** on 21 November 2026, including registered `nn_training_gradient_control` (17 NN-labelled tests registered). The detailed CTest output and separate Release/Metal/Doxygen checks were not supplied.
- [ ] **Deferred to E.07:** full rollback when an Adam proposal fails *after successful gradient control*; cross-component step commit and cumulative metrics must be tied to the training session, not claimed by E.05.

##### NN.1.07E.06 — AdamFP64 serialisable state (implementation committed; CPU validation pending)
- [x] Add Doxygen-documented `AdamSnapshot` v1 with successful step count, Adam learning rate/beta1/beta2/epsilon, stable ordered parameter IDs and both first/second FP64 moment tensors. Existing 5-argument AdamFP64 construction remains compatible; optional explicit parameter IDs protect equal-shaped tensors from accidental reorder.
- [x] Add `AdamFP64::snapshot()` and transactional `restore()`: validate version, exact hyperparameter identity, ordered ID identity, tensor count/shape, finite model state/moments, nonnegative second moments, and then commit staged moment tensors and step count via noexcept move. Model parameter values and gradients remain unchanged by restore.
- [x] Add independent `nn_adam_snapshot` CTest: 10 continuous steps vs 5 + in-memory state/model-value restore + 5 steps, exact FP64 parameter/moment parity, invalid schema/hyperparameters/IDs/order/moments/shapes and rejection without mutation.
- [ ] **Validation gate:** user runs `bash scripts/test.sh --nn` and `bash scripts/test.sh --all`. Expected **18 NN and 26 full CPU** CTests; results not yet received.
- [ ] **E.08 boundary:** snapshot is an in-memory state object, not a file codec. Its stored ordered IDs must be supplied by the training session and checked against model state. Full checkpoint payload, integrity, atomic disk writes, RNG and dataset cursor remain E.08.
- [ ] **E.07 boundary:** complete transactional rollback if a later optimiser update follows clipping or accumulation remains a separate training-session responsibility.

##### NN.1.07E.07 — Training session state machine and failure boundaries
- [ ] Implement explicit Created/Ready/Accumulating/Updating/Validating/Checkpointing/Failed/Completed states with guarded transitions.
- [ ] Define and test whole-window rollback/fail-closed handling across task, gradients, controller, Adam, cursor and RNG streams.
- [ ] Enforce that only successful Adam commits increment successful steps; explicitly test failure injection at each phase.
- [ ] Implement deterministic validation cadence that leaves train parameters/gradients, optimiser moments, train cursor and RNG state unchanged.

##### NN.1.07E.08 — Versioned checkpoint codec and filesystem atomicity
- [ ] Store exact stable parameter IDs/shapes/values, both Adam moments, step, canonical policy, run/task identifiers, dataset digest, code revision, metric ID, full RNG engine states, data cursor, counters and best validation metric.
- [ ] Checkpoint only at a committed optimiser boundary with zero pending gradients for v1; reject incompatible/partial states.
- [ ] Implement bounded and integrity-checked canonical FP64 serialization with schema/version/shape/truncation/non-finite validation.
- [ ] Save via same-directory temporary file, flush/sync and atomic rename; retain previous valid checkpoint and test interruption/corruption.
- [ ] Restore with parse/validate/stage/commit and failure atomicity; test task/dataset/parameter/policy/metric mismatches.

##### NN.1.07E.09 — Deterministic restart proof
- [ ] Compare 10 continuous optimiser steps with 5 + checkpoint/save/load + 5 steps using identical ordered synthetic image batches.
- [ ] Require bit-identical (where same backend deterministic operations permit) parameter tensors, Adam moments, step counters, cursor and RNG state; otherwise record justified FP64 tolerance.
- [ ] Test interrupted-window replay from last committed checkpoint and correct handling of resumed validation cadence.

##### NN.1.07E.10 — Fault-injection regression
- [ ] Fail deliberately during batch fetch, forward, backward, loss validation, gradient constraint, Adam update, checkpoint write and checkpoint load.
- [ ] Verify no false successful-step count, no partially committed parameter or optimiser state and no incorrectly advanced persisted cursor.
- [ ] Verify failed checkpoint write preserves the previously committed checkpoint and reports failure distinctly.

##### NN.1.07E.11 — Build, documentation and evidence
- [ ] Register new production files in `src/nn/training/CMakeLists.txt` and each CTest in `tests/nn/CMakeLists.txt` without duplicate compilation.
- [ ] Document all new public C++ APIs with Doxygen and check the optional docs build.
- [ ] Run `bash scripts/test.sh --nn`, `--p0` and `--all`, then a clean Release build; separately validate Metal if supported.
- [ ] Add exact observed test output, commit SHA and reproducibility metadata to checklist, STATUS and CHANGELOG only after evidence.

##### NN.1.07E.12 — Image-first demonstration and handoff
- [ ] Train the deterministic toy reconstruction fixture and demonstrate decreasing held-out MSE without train/val leakage.
- [ ] Produce an exact reproducibility report with dataset synthetic fixture, seed, policy, code revision, optimiser and checkpoint file hashes.
- [ ] Hand off model architecture, image decoding/real datasets and visual reconstruction to AE tickets; hand off epsilon prediction to NN.2–NN.3.
- [ ] Reserve rank-five temporal video tensors and compute/parameter scaling decisions for a future video ticket; make no video capability claim now.

**Exit criteria:** All E.01–E.12 gates relevant to the approved v1 scope are reviewed, implemented and user-validated. This milestone delivers image-training infrastructure, **not** a complete generative image model.

#### NN.1.07F — Reproducible training and validation experiments
- [ ] Demonstrate deterministic loss reduction on a small controlled mapping as a **smoke test only**.
- [ ] Define fixed data splits, baseline initialisation, validation schedule, early stopping and model promotion criteria.
- [ ] Compare unclipped, Euclidean-clipped and metric-radial-clipped optimisation under matched experimental conditions.
- [ ] Measure training/validation loss, step rejection, clipping frequency, gradient/update norms, computational cost and reproducibility.
- [ ] Do not claim autoencoder or diffusion training complete until their separate dataset/quality milestones pass.

#### NN.1.07G — Experimental optimiser-update trust region
- [ ] Specify an inspectable, pre-commit AdamFP64 proposed update; preserve the current validated atomic-update path.
- [ ] Define a separate positive-definite parameter-space metric and enforce `delta_theta^T M_theta delta_theta <= r^2` on the **actual proposed displacement**.
- [ ] Document handling of Adam moment-state updates, rejection versus rescaling and rollback/transaction semantics before implementation.
- [ ] Test trust-region boundaries, non-finite proposals, multi-parameter atomicity and deterministic replay.
- [ ] Compare no constraints, gradient clipping, update trust regions and both controls; treat this mode as experimental until validated.

**Dependencies:** NN.1.07A precedes B–G. NN.1.07C is the reference for D. NN.1.07G requires an explicit extension of AdamFP64's proposal/commit interface; gradient clipping alone does not bound Adam's actual parameter step.

**Sign-off:** Each ticket remains unchecked until its design/implementation is reviewed and independently validated. Record CPU, Doxygen, Release and Metal evidence separately. Related autoencoder training and checkpoint milestones are in [AE.5–AE.6](AUTOENCODER_CHECKLIST.md); epsilon predictor training and sampler integration remain in NN.2–NN.3.

**NN.1.02 — FP64 Linear Layer (signed off 20 November 2026)**

- [x] Implement rank-1 and batched rank-2 `Linear` forward propagation (`y = Wx + b`).
- [x] Implement input, weight and bias backward gradients.
- [x] Validate deterministic known answers, batch shapes and gradients.
- [x] Validate finite-difference numerical gradients, accumulation and `zero_grad()`.
- [x] Register `test_nn_linear` under `nn;core` and add public Doxygen API comments.

**Validation:** User reports the NN, P0 and full CPU test suites passed after NN.1.02. Source, tests and CMake wiring verified on GitHub. Exact counts, durations and Doxygen build output not supplied; Release, Metal and clean-clone unverified. The combined activation-and-linear NN.1 item remains open until activation is implemented.

**NN.1.01 validation (20 November 2026, macOS CPU Debug):** User confirmed `bash scripts/test.sh --nn`, `--p0` and `--all` passed. The Layer/Parameter headers, `test_nn_layer` registration and `--nn` runner option were verified on GitHub. Counts, times, Release and Metal validation were not supplied; NN.1 remains in progress.

## NN.2 — Spatial epsilon-prediction network (Roadmap Milestone 5)

- [ ] Define `predict_epsilon(z_t, t)` API returning the same Tensor shape.
- [ ] Establish a simple spatial CNN denoiser baseline (not NLP vector reshaping).
- [ ] Implement sinusoidal timestep embedding and conditioning.
- [ ] Implement residual blocks and output projection.
- [ ] Validate forward shapes for 3D and 4D latent Tensors.
- [ ] Validate backpropagation through all denoiser parameters.
- [ ] Test zero/invalid/NaN/Inf inputs and timestep boundaries.
- [ ] Use MSE between predicted epsilon and the known injected noise.
- [ ] Demonstrate seeded learning on a small synthetic latent dataset.
- [ ] Demonstrate validation loss improvement over an untrained baseline.
- [ ] Add Doxygen API and numerical-validation evidence.

**Exit criterion:** The network produces a shape-correct noise prediction and
can be trained to reduce held-out epsilon-prediction error.

## NN.3 — Training and sampler integration (Roadmap Milestones 6–7)

- [ ] Define the boundary between `Sampler::sample_step` and the denoiser.
- [ ] Build training samples using `q_sample(z0, t, epsilon)`.
- [ ] Train on known injected Gaussian noise without placeholder predictions.
- [ ] Implement/checkpoint model and optimiser state, RNG seed and NN.1.07A policy/configuration via the NN.1.07E shared training lifecycle.
- [ ] Test deterministic checkpoint resume.
- [ ] Add a reverse loop calling the trained denoiser at each timestep.
- [ ] Verify reverse-step ordering and non-finite trajectory guards.
- [ ] Integrate trained latent autoencoder decoder (Roadmap Milestone 4; see [AE.8](AUTOENCODER_CHECKLIST.md)).
- [ ] Demonstrate seeded end-to-end output from trained model checkpoints.
- [ ] Record CPU numerical and regression evidence; Metal separately.

**Exit criterion:** The sampler uses actual trained predictions and the pipeline
can decode generated latents into images.

## Explicit boundary

Completion of P0.9 means **one validated stochastic reverse step**, not
completion of NN.1–NN.3 and not image generation. No placeholder/random
predicted epsilon is accepted as a substitute for the neural network.

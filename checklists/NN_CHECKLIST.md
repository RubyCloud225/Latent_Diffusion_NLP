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

#### NN.1.07C — Gradient geometry interface and Euclidean reference
- [ ] Implement a gradient measurement and constraint interface for registered `Parameter` tensors; measure the global norm without altering gradients.
- [ ] Implement FP64 Euclidean global-norm radial clipping as an opt-in baseline, with zero-norm and non-finite safeguards.
- [ ] Validate direction preservation, threshold boundary, no-op behaviour, deterministic results and no partial mutation on error.
- [ ] Log pre/post gradient norms, clipping scale, clipping frequency and rejected steps.

#### NN.1.07D — Diagonal and block-metric gradient controller
- [ ] Define and validate positive-definite diagonal/block metrics, parameter ownership/order, weights and metric versioning.
- [ ] Implement opt-in metric-norm radial scaling using `sqrt(g^T M g)` with FP64 overflow/underflow protection.
- [ ] Document that radial scaling is **not generally the Euclidean closest-point projection onto an ellipsoid**.
- [ ] Compare with the Euclidean baseline using identical parameters, batches, RNG seeds and optimisation settings.
- [ ] Keep approximate Fisher/natural-gradient methods out of scope until separately designed and empirically justified.

#### NN.1.07E — Training execution lifecycle and checkpoints
- [ ] Define batch/epoch/step state transitions: zero_grad, forward, objective, backward, gradient controller, optimiser, validation and checkpoint.
- [ ] Ensure invalid loss/gradients/metrics/updates reject an entire step without partially changing model or optimiser state.
- [ ] Store model parameters, optimiser moments and step count, policy/schema version, data manifest, RNG state and code revision.
- [ ] Test checkpoint integrity, incompatible-version rejection, deterministic resume and interrupted-step recovery.
- [ ] Keep dataset-specific autoencoder and denoiser objectives in their own tracks; use this shared execution policy for both.

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

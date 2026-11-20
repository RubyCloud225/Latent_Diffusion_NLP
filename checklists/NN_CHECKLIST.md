# Neural Network and Epsilon Predictor Checklist

This is a separate implementation track from [P0 — Production Core](P0_CHECKLIST.md).
P0.8 and P0.9 accept **externally supplied predicted epsilon**: they implement
mathematically correct reverse steps, not a trained model. An actual learned
epsilon predictor is required before end-to-end image generation.

Aligned with [ROADMAP.md](../ROADMAP.md), especially Milestones 3, 4, 5, 6 and 7. Image encoding, decoding, reconstruction training and latent handoff are separately tracked in [AUTOENCODER_CHECKLIST.md](AUTOENCODER_CHECKLIST.md).

## NN.1 — Trainable neural operations (Roadmap Milestone 3)

- [x] Define a production `Layer` interface with shape contracts.
- [ ] Implement or port Conv2D with forward and backward propagation.
- [ ] Implement activation and linear layers with correct backward passes.
- [ ] Implement spatial upsampling/transposed convolution as required.
- [x] Define parameter, gradient and zero-grad interfaces.
- [ ] Integrate Adam with model parameters and deterministic initialisation.
- [ ] Add gradient clipping and finite-value checks.
- [ ] Test layer shapes, invalid inputs and input immutability as applicable.
- [ ] Validate analytical gradients against finite differences.
- [ ] Demonstrate loss reduction on a deterministic toy mapping.
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

#### NN.1.06 — FP64 Adam
- [ ] Adapt or replace isolated FP32 Adam to work with production FP64 Parameters.
- [ ] Add seeded/known-answer, optimiser-state and validation tests.

#### NN.1.07 — Initialisation and training
- [ ] Implement reproducible nonzero parameter initialisation and gradient clipping.
- [ ] Show a deterministic tiny NN learns a toy mapping and add `nn_training` test.

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
- [ ] Implement/checkpoint model and optimiser state, seed and configuration.
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

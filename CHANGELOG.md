# CHANGELOG.md

# Changelog

All notable changes to the production image-generation line of this project will be documented in this file.

The project began as an experimental latent-diffusion NLP research implementation. The changelog below establishes a clean baseline for the transition into a production-oriented image generator.

Format is based broadly on Keep a Changelog principles, while version numbers reflect the internal development roadmap.

---

## [Unreleased]

### NN.1.07 training-policy and geometric-control ticket redesign (20 November 2026)

- Replaced the original combined initialisation/clipping/toy-training ticket with NN.1.07A–G: policy schema, seeded initialisation, Euclidean reference clipping, diagonal/block metric-radial clipping, training/checkpoint lifecycle, controlled training evaluations and experimental Adam update trust region.
- Explicitly separated raw-gradient magnitude constraints from constraints on actual proposed Adam parameter displacements; clipping remains opt-in and all geometric methods require independent validation.
- Aligned AE.5/AE.6 and NN.3 checklist dependencies with the shared policy and checkpoint framework; updated high-level NN.1 completed foundations.
- Planning/documentation only. No training implementation or additional test outcome is claimed.


### NN.1.06 AdamFP64 production optimiser (20 November 2026, CPU validated)

- Verified new `latent/nn/AdamFP64.hpp`, `src/nn/AdamFP64.cpp`, and `tests/nn/test_adam_fp64.cpp` integrated with `latent_core` and registered as `nn_adam_fp64` in CMake.
- Production FP64 Adam uses non-owning `Parameter*` registration, bias-corrected moments and staged, validated updates. Existing experimental FP32 Adam remains separately maintained.
- Tests cover first/multiple update steps, deterministic replay, multiple tensors, reset, zero-grad, bad inputs, atomic failed-step recovery and Linear integration.
- User reports 16/16 full CPU suite PASS. Exact test logs/timings, Doxygen generation, Release, Metal and clean-clone results were not supplied.


### NN.1.05 Conv2D backward (20 November 2026, CPU tests passed)

- Verified committed NCHW FP64 backward propagation for input, kernel and bias gradients, cached forward input, and accumulation into production Parameters.
- Independent `nn_conv2d_backward` tests cover known answers, batched accumulation, zero_grad(), numerical finite differences and invalid-gradient rejection. CMake registers the test under `nn`.
- User reports NN, P0 and full CPU suites all passed; registered counts expected 7 NN, 7 P0 and 15 overall, with exact output/timings not provided. Doxygen generation, Release and Metal remain unverified.


### NN.1.04 Conv2D forward (20 November 2026, CPU Debug validated)

- Added FP64 NCHW Conv2D reference forward implementation, supporting batch/channel dimensions, stride, symmetric padding, learnable kernel and bias storage; added independent `nn_conv2d_forward` CTest coverage.
- Fixed misnamed header `latent/nn/Cov2D.hpp` to `latent/nn/Conv2D.hpp` after compilation detected the wrong filename.
- User reports `--nn`, `--p0`, and `--all` CPU test suites pass after the correction; registered counts expected 6 NN, 7 P0 and 14 total (actual counts/timings not supplied).
- Backward propagation was subsequently added in NN.1.05. Doxygen generation, Release and Metal are unverified.


### NN.1.03C Activation numerical validation (20 November 2026, CPU validated)

- Added `tests/nn/test_activation_validation.cpp`, independently checking ReLU and SiLU backward gradients with central finite differences (excluding the nondifferentiable point for ReLU).
- Added boundary and extreme-value checks, rank-4 shape tests and tests proving invalid forward inputs do not corrupt cached valid state.
- Registered `nn_activation_validation` under the `nn` CTest label; production activation code unchanged.
- User reports full CPU regression suite PASS (13/13), including registered activation numerical-validation test. Individual test timings, Doxygen generation, Release and Metal not independently verified.

### NN.1.03B SiLU activation (20 November 2026, CPU Debug validated)

- Added Doxygen-documented FP64 SiLU (Swish) forward/backward to existing `Activation.hpp` and `Activation.cpp`, using overflow-safe sigmoid branches.
- Added `tests/nn/test_silu.cpp` to check known answers, extreme values, tensor shapes, non-finite inputs and invalid backwards calls.
- Registered `nn_silu` in CMake with CTest label `nn`; existing ReLU behaviour retained.
- User reports `--nn`, `--p0` and `--all` CPU test suites all passed following SiLU integration; expected registered counts 4 NN / 7 P0 / 12 total, exact logs/timing not supplied. Doxygen generation, Release and Metal results not independently verified.

### NN.1.03A ReLU activation (20 November 2026, CPU Debug validated)

- Added Doxygen-documented FP64 ReLU implementation in `latent/nn/Activation.hpp` and `src/nn/Activation.cpp`, using cached forward inputs and a zero derivative at x=0.
- Added `tests/nn/test_relu.cpp` covering forward/backward known answers, Tensor shapes, immutability, parameterlessness and invalid input/gradient cases.
- Registered `nn_relu` under the `nn` CTest label, linked activation to `latent_core`, and included `latent/nn` in the optional Doxygen target.
- Split NN.1.03A–NN.1.07 into individual checklist tickets. User confirms `--nn`, `--p0`, and `--all` CPU Debug test suites passed following ReLU integration; exact test logs and durations not provided. Release, Metal and Doxygen generation are unverified.

### NN.1.02 FP64 Linear Layer sign-off (20 November 2026)

- Verified committed `Linear.hpp`, `Linear.cpp`, `test_linear.cpp` and `latent_core`/CTest integration on GitHub.
- User reports NN, P0 and full CPU regressions passed following implementation; exact counts and timings not supplied for this run.
- Linear tests cover rank-1 and batched forward propagation, backpropagation, gradients of weights/bias/inputs, finite differences, accumulation, zero-grad and validation.
- Doxygen source comments present; generated documentation, Release and Metal not independently verified.

### NN.1.01 Layer and Parameter interfaces (20 November 2026)

- Verified production FP64 `Layer` and `Parameter` interfaces, test-only ScaleLayer coverage, `test_nn_layer` CMake registration (`nn;core`) and `scripts/test.sh --nn` support on GitHub.
- User confirmed macOS Debug CPU results: `bash scripts/test.sh --nn` PASS, `--p0` PASS and `--all` PASS. Exact counts and timings not supplied for this run; no Release or Metal validation claimed.
- Signed off the Layer interface and parameter/gradient/zero-grad interfaces only. NN.1 Conv2D, Linear, activation, optimiser integration and full training remain pending.

### Documentation workflow and contributor guidance (20 November 2026)

- Added `scripts/run_documentation.sh` for optional Doxygen generation in isolated `build-docs/` with `--open`, `--help`, and `DOCS_BUILD_DIR` override.
- Added root `COMMANDS.md` with supported build, CTest, documentation and Git commands.
- Added root `CONTRIBUTION.md` outlining licensing, API documentation, numerical test standards and pull-request workflow.
- Updated README and `docs/DOXYGEN.md` with the canonical script; ignored documentation build output in `.gitignore`.
- Documentation script/HTML generation still require execution on a host with Doxygen installed; not independently validated here.

### P0.8 reverse diffusion and Doxygen (20 November 2026)

- Added deterministic reverse diffusion coefficients, posterior variance, clean-latent reconstruction and reverse mean to the existing production GaussianDiffusion class.
- Added `reverse_diffusion` CTest target (`p0;diffusion`) covering analytical coefficients, known answers, t=0, shape checks, finiteness, deterministic repeatability and forward/reverse consistency.
- Added Doxygen-compatible public API comments to Tensor, RandomGenerator, NoiseSchedule and GaussianDiffusion headers.
- Added optional `BUILD_DOCS=ON` CMake `docs` target and `docs/DOXYGEN.md` instructions; normal builds do not require Doxygen.
- User confirmed macOS Debug P0 and full CPU regression suites passed following P0.8 implementation (expected registrations: 7 P0, 8 total). Exact output and timings were not provided. P0.8 CPU sign-off recorded; documentation generation, Release, Metal and clean-clone validation remain outstanding.

### P0.7 Forward Diffusion implementation (20 November 2026)

- Implemented `GaussianDiffusion::q_sample` in the production `latent_core` library using P0.6 precomputed cumulative-alpha square-root coefficients and caller-supplied noise.
- Added timestep bounds, exact-shape and finite-value checks; rejects non-finite results; preserves input and noise Tensors.
- Added `forward_diffusion` CTest target labelled `p0;diffusion`, with known-answer, zero-noise, zero-input, shape, determinism, boundary and invalid-input tests.
- macOS Debug CPU validation (20 November 2026), user-provided output: `bash scripts/test.sh --p0` PASS (6/6, 0.53 s); `bash scripts/test.sh --all` PASS (7/7, 0.03 s), including the new forward-diffusion test and Clifford regression. Release, Metal and clean-clone remain unverified.

### P0.6 Noise Schedule validation (20 November 2026)

- Verified pushed `NoiseSchedule.hpp`, `NoiseSchedule.cpp`, `test_noise_schedule.cpp`, and CMake test registration.
- User-reported macOS Debug: `bash scripts/test.sh --p0` PASS (5/5, 0.41 s); `bash scripts/test.sh --all` PASS (6/6, 0.03 s).
- Tests cover beta interpolation, cumulative alpha, square-root coefficients, finite values, invalid configurations, timestep bounds, and deterministic construction.
- Release, Metal and clean-clone validation remain outstanding.

### P0.5 local validation (20 November 2026)

- User-reported macOS Debug run: `bash scripts/test.sh --p0` PASS (4/4; 0.28 s), including `random`.
- User-reported full CPU regression: `bash scripts/test.sh --all` PASS (5/5; 0.02 s), including `tensor`, `fp16`, `diffusion_harness`, `clifford_projection`, and `random`.
- **Repository sync pending:** as checked on GitHub, `latent/core/Random.hpp` and `src/core/Random.cpp` are still empty placeholders, `tests/core/test_random.cpp` is not committed, and the default-branch CMake does not register `test_random`. The passing implementation exists in the user's local working tree and must be committed/pushed before P0.5 repository sign-off.
- Metal, Release, and clean-clone validation not performed in this run.

### P0.4 CMake fix (20 November 2026)

- Created the missing production `latent_core` static-library target with `src/core/Tensor.cpp`.
- Removed `Tensor.cpp` from `latent_experimental` so the production and experimental libraries remain separate.
- Added the repository-root public include directory, matching the existing `latent/core/Tensor.hpp` header location.
- Linked `test_tensor` to `latent_core` and retained its `p0;core` CTest labels.
- Validated on macOS Debug (20 November 2026): `bash scripts/test.sh --p0` PASS (3/3, 0.16 s) and `bash scripts/test.sh --all` PASS (4/4, 0.02 s); `tensor`, FP16, diffusion harness and experimental Clifford projection passed. Release, Metal, and clean clone not tested.


### P0.3 test harness (20 November 2026)

- Fixed CPU-only script builds to default `BUILD_METAL_TESTS=OFF` explicitly, preventing cached or Apple-default Metal targets from breaking P0 validation when the optional Metal toolchain is absent. The `--metal` test selection enables Metal explicitly.
- Observed user's C++ CPU targets build, but the optional shader compilation failed because the Apple Metal toolchain was missing; P0 CTest subsequently passed (2/2), and full CPU regression passed (3/3).

- Registered existing FP16 test as `p0;core` in CTest.
- Added a diffusion reference-fixture executable under `tests/diffusion` and registered it as `p0;diffusion`.
- Labelled Clifford tests `experimental` and Apple Metal tests `experimental;metal` to keep them out of the P0-only selection.
- Added `ctest --test-dir build -L p0 --output-on-failure` as the P0-only test invocation.
- Implemented existing `scripts/build.sh` and extended `scripts/test.sh` to select `--all`, `--p0`, `--core`, `--diffusion`, `--experimental`, or `--metal`; `BUILD_TYPE`, `BUILD_DIR`, and `BUILD_METAL_TESTS` are configurable.
- Reference fixtures are **not** production diffusion API validation; tests for the actual implementation remain P0.6–P0.9.
- macOS Debug validation (20 November 2026): `bash scripts/test.sh --p0` PASS (2/2, 0.27 s); `bash scripts/test.sh --all` PASS (3/3, 0.13 s). Metal excluded because its optional compiler is unavailable. Clean-clone, Release and full production diffusion validation remain outstanding. The checked-in CMake also lacks the `latent_core` target ticked under P0.2, which needs separate reconciliation.

### Added

- Production image-generation architecture.
- `DESIGN.md`.
- `ROADMAP.md`.
- `STATUS.md`.
- Initial productionisation release plan.
- Planned separation between:
  - tensor core;
  - diffusion mathematics;
  - neural layers;
  - image encoder/decoder;
  - denoiser;
  - checkpointing;
  - training;
  - inference.
- Defined first image-generation success criterion:
  - initialise latent Gaussian noise;
  - perform learned reverse diffusion;
  - decode final latent;
  - save a recognisable image;
  - reproduce generation from a fixed seed.
- Defined conventional autoencoder latent path as the initial baseline.
- Defined Clifford latent representation as an optional experimental codec rather than a mandatory production dependency.
- Defined Apple Metal as the first planned accelerator backend following CPU validation.

### Changed

- Project direction expanded from latent-diffusion NLP research toward a general latent-diffusion engine with image generation as the primary production target.
- Development strategy now prioritises mathematical correctness, deterministic execution, testing, and modularity before model scale.
- Existing NLP functionality is designated for migration into `legacy/nlp/`.
- Existing experimental diffusion implementations are designated for consolidation into one production diffusion core.
- Image denoising will initially use noise-prediction MSE rather than the earlier NLL-focused research training path.
- Future image denoiser will operate on spatial latent tensors rather than reshaping flat NLP vectors into artificial image grids.
- Generated build artefacts will no longer be retained in source control.

### Planned

- Repository cleanup.
- CMake restructuring.
- Core library target.
- Test executable.
- Contiguous tensor abstraction.
- Seeded RNG abstraction.
- Unified beta/alpha/alpha-bar schedule.
- Correct forward `q_sample`.
- Correct learned reverse sampling.
- Image loading and saving.
- Image autoencoder.
- Spatial epsilon predictor.
- Full checkpointing.
- Generator CLI.

### Added

- Defined production Clifford latent architecture.
- Defined `Cl(3,0)` eight-coefficient multivector representation.
- Defined trainable `64 → 8` latent projection.
- Defined projection matrix as a differentiable model parameter.
- Defined CPU implementation as the numerical reference.
- Defined fixed orthonormal known-answer projection for testing.
- Defined CPU/Metal parity testing strategy.
- Defined `FP16.hpp` as shared CPU FP16 conversion support.
- Defined Metal-native `half` output path.

### Changed

- Replaced the planned truncation-based Clifford compression behaviour with an explicit trainable projection:

\[
c = Px.
\]

- Clifford compressed objects will no longer retain an unnecessary full copy of the original source latent.
- Production projection weights will be learned during training.
- Tests will use fixed deterministic orthonormal weights rather than trained or random weights.
- Metal acceleration will implement the same projection mathematics as the CPU reference rather than a separate GPU-specific approximation.

### Deprecated

The original research implementation:

```cpp
int k = std::min<int>(8, input.size());

for (...) {
    output[i] = input[i];
}
```

is now considered legacy prototype behaviour and will not form part of the production Clifford latent path.

---

## [0.0-research] — Legacy Baseline

### Added

- Pure C++ latent-diffusion research pipeline.
- Basic BPE tokenizer.
- Deterministic token embedding.
- Experimental Clifford-manifold compression.
- FP16 conversion helpers.
- Huffman serialization experiment.
- Beta schedule implementation.
- Gaussian diffusion experiments.
- Reverse-process experiments.
- Normal-distribution probability utilities.
- Custom Adam optimiser.
- Custom convolution layer.
- Pooling layer.
- ReLU layer.
- flatten layer.
- fully connected layer.
- CNN epsilon-predictor experiments.
- sinusoidal timestep embedding.
- diffusion training experiments.
- human-readable training logs.
- partial binary checkpointing.

### Fixed during research iteration

- Convolution layer weight initialisation corrected so independent layers do not unintentionally share identical initial values.
- Excessive inner-loop convolution debug output removed.
- Pooling tensor dimension ordering corrected.
- Pooling allocation behaviour corrected.
- Fully connected layer corrected to perform weighted projection.
- Neural-network stage execution ordering corrected.
- Epsilon predictor changed to persistent model state rather than recreating weights during prediction.
- Epsilon output changed to remain unbounded rather than being passed through ReLU.
- Timestep information added to epsilon prediction.

### Known limitations

- Repository architecture remains research-oriented.
- Build artefacts are currently tracked.
- Maintained neural-network implementation is not fully represented in the root CMake target.
- Multiple diffusion formulations coexist.
- Reverse diffusion still contains placeholder random epsilon behaviour.
- Full denoiser backpropagation is incomplete.
- Checkpointing does not persist all trainable layers.
- Image encoder and decoder do not yet exist as production modules.
- No complete image-generation path exists.

---

## Version Plan

### [0.1.0] — Production Core

Target:

- repository cleanup;
- modern CMake;
- tests;
- tensor core;
- deterministic RNG;
- validated diffusion schedule;
- `q_sample`;
- reverse-process mathematical foundation.

### [0.2.0] — Image Autoencoder

Target:

- image I/O;
- encoder;
- decoder;
- reconstruction training;
- complete autoencoder checkpoint.

### [0.3.0] — Latent Denoiser

Target:

- spatial epsilon predictor;
- timestep conditioning;
- noise-prediction training;
- validated denoising loss.

### [0.4.0] — First Generator

Target:

- Gaussian latent initialisation;
- learned reverse diffusion loop;
- latent decoding;
- seeded image generation;
- `latent_generate` executable.

### [0.5.0] — Experimental Latent Geometry

Target:

- Clifford latent codec;
- wavelet experiments;
- baseline comparisons.

### [0.6.0] — Conditioning

Target:

- conditioning interface;
- class or image conditioning;
- guidance.

### [0.7.0] — Text-to-Image

Target:

- text encoding;
- prompt conditioning;
- text-guided image generation.

### [0.8.0] — Metal Backend

Target:

- backend abstraction;
- Metal tensor kernels;
- accelerated inference/training;
- CPU/Metal validation.

### [1.0.0] — Production Image Generator

Target:

- reproducible image generation;
- complete checkpointing;
- validated training and inference;
- stable CLI;
- documented configuration;
- automated tests;
- hardware-aware execution;
- extensible model architecture.

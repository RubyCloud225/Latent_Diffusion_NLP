# ROADMAP.md

# Latent Diffusion Image Generator — Roadmap

## Objective

Transform the current experimental latent-diffusion NLP repository into a production-oriented C++ image-generation engine.

The roadmap deliberately proves the system in increasing levels of complexity:

```text
correct maths
    ↓
clean numerical core
    ↓
image reconstruction
    ↓
latent denoising
    ↓
unconditional image generation
    ↓
conditioning
    ↓
hardware acceleration
    ↓
scale
```

---

# Milestone 0 — Repository Rehabilitation

## Goal

Create a clean, buildable, testable foundation before extending model capability.

## Tasks

- [x] Add `DESIGN.md`.
- [x] Add `ROADMAP.md`.
- [x] Add `STATUS.md`.
- [x] Reset `CHANGELOG.md` around the productionisation effort.
- [x] Expand `.gitignore`.
- [x] Remove generated `build/` artefacts from source control.
- [x] Separate legacy NLP code from production code.
- [x] Introduce `include/`, `src/`, `apps/`, and `tests/`.
- [ ] Update CMake target structure.
- [ ] Build the current maintained neural-network implementation.
- [ ] Add compiler warnings.
- [ ] Add debug and release build modes.
- [ ] Add a minimal automated test executable.
- [ ] Establish formatting and naming conventions.

## Exit criteria

A fresh clone can be configured, built, and tested without committed build artefacts or hidden local state.

---

# Milestone 1 — Production Diffusion Core

## Goal

Replace overlapping diffusion implementations with one mathematically consistent and deterministic core.

## Tasks

- [ ] Introduce `NoiseSchedule`.
- [ ] Precompute beta values.
- [ ] Precompute alpha values.
- [ ] Precompute cumulative alpha-bar values.
- [ ] Implement correct `q_sample`.
- [ ] Accept externally supplied Gaussian noise.
- [ ] Add seeded RNG abstraction.
- [ ] Remove locally default-constructed random engines from diffusion operations.
- [ ] Implement posterior coefficients required for reverse sampling.
- [ ] Implement one reverse-step API.
- [ ] Remove placeholder reverse epsilon sampling from production code.
- [ ] Separate sampling from schedule mathematics.
- [ ] Add linear schedule.
- [ ] Add diffusion unit tests.
- [ ] Add finite-value and boundary checks.

## Exit criteria

For a given input tensor, timestep, noise tensor and seed, the diffusion core produces deterministic, mathematically validated output.

---

# Milestone 2 — Tensor and Image Infrastructure

## Goal

Move from NLP vectors to spatial image tensors.

## Tasks

- [ ] Implement contiguous `Tensor`.
- [ ] Support 1D, 2D, 3D and 4D shapes.
- [ ] Add shape validation.
- [ ] Add contiguous indexing.
- [ ] Add tensor initialisation helpers.
- [ ] Add normal/Gaussian tensor generation.
- [ ] Add image representation.
- [ ] Add minimal image reader.
- [ ] Add minimal image writer.
- [ ] Add normalisation to `[-1, 1]`.
- [ ] Add denormalisation.
- [ ] Add resize or fixed-size validation.
- [ ] Add image round-trip tests.

## Initial target

```text
32 × 32 RGB
```

A later milestone may move to:

```text
64 × 64 RGB
```

## Exit criteria

The executable can load an image, convert it to a tensor, normalise it, restore it, and save a visually equivalent image.

---

# Milestone 3 — Trainable Neural Core

## Goal

Provide reliable forward and backward computation for the small set of operations needed by the image model.

## Tasks

- [ ] Standardise `Layer` interface.
- [ ] Implement or rehabilitate `Conv2D`.
- [ ] Implement activation layer.
- [ ] Implement linear layer.
- [ ] Implement upsample or transposed convolution.
- [ ] Implement backward pass for each trainable layer.
- [ ] Expose trainable parameters consistently.
- [ ] Add gradient buffers.
- [ ] Add zero-grad operation.
- [ ] Integrate Adam with model parameters.
- [ ] Add gradient clipping.
- [ ] Add numerical gradient tests.
- [ ] Validate parameter updates reduce synthetic losses.

## Exit criteria

A small neural network can learn a deterministic toy mapping and pass numerical gradient checks.

---

## Milestone 3A — Trainable Clifford Projection

### Goal

Replace the original truncation-based Clifford compression prototype with a mathematically explicit, trainable latent projection.

The production model will map:

\[
\mathbb{R}^{64}
\rightarrow
Cl(3,0)
\]

using a trainable \(8 \times 64\) projection matrix.

### Tasks

- [ ] Add `FP16.hpp`.
- [ ] Add `CliffordMultivector.hpp`.
- [ ] Define fixed eight-blade `Cl(3,0)` coefficient ordering.
- [ ] Add `CliffordProjection.hpp`.
- [ ] Implement CPU `64 → 8` projection.
- [ ] Expose projection matrix as trainable parameters.
- [ ] Implement projection backward pass.
- [ ] Compute gradients with respect to projection weights.
- [ ] Compute gradients with respect to source latent.
- [ ] Add FP32 Clifford coefficient representation.
- [ ] Add FP16 coefficient conversion.
- [ ] Remove source-latent duplication from compressed multivector storage.
- [ ] Add deterministic orthonormal test projection.
- [ ] Add known-answer CPU test.
- [ ] Add malformed-input tests.
- [ ] Add finite-value tests.

### Known-answer test

Use a fixed orthonormal projection:

```text
P[0,0] = 1
P[1,1] = 1
...
P[7,7] = 1
```

For:

```text
input[i] = i + 1
```

expect:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

### Exit criteria

The CPU implementation:

- produces the exact expected coefficients for the fixed test projection;
- supports trainable projection parameters;
- propagates gradients;
- produces valid FP16 representations;
- passes numerical gradient validation.

---

## Milestone 3B — Clifford Metal Backend

### Goal

Implement the same Clifford projection using Apple Metal.

### Tasks

- [ ] Add `CliffordProjection.metal`.
- [ ] Add `CliffordProjectionMetal.mm`.
- [ ] Upload 64D source latent to Metal buffer.
- [ ] Upload projection matrix to Metal buffer.
- [ ] Dispatch eight output threads.
- [ ] Compute one Clifford blade coefficient per thread.
- [ ] Produce FP32 output.
- [ ] Produce native Metal `half` output.
- [ ] Add CPU/Metal parity test.
- [ ] Validate fixed projection known-answer result.
- [ ] Validate random projection CPU/Metal agreement.
- [ ] Add invalid-buffer/error handling.
- [ ] Benchmark projection execution.

### Initial validation path

```text
CPU input
   ↓
Metal buffers
   ↓
Metal Clifford projection
   ↓
copy coefficients to host
   ↓
compare with CPU reference
```

### Production path

After parity is established:

```text
GPU image latent
      ↓
GPU Clifford projection
      ↓
GPU diffusion
      ↓
GPU denoiser
```

No unnecessary host round-trip should occur in the final training path.

### Exit criteria

Metal output matches CPU reference output within defined FP32/FP16 numerical tolerances.

---

## Milestone 9 — Clifford Latent Evaluation

### Goal

Evaluate the learned Clifford projection as an image-latent representation rather than assuming it improves the model.

### Baseline

Compare:

```text
standard learned latent
```

against:

```text
standard latent
      ↓
trainable Clifford projection
      ↓
Cl(3,0) latent
```

### Measurements

- [ ] reconstruction error;
- [ ] diffusion denoising loss;
- [ ] generation quality;
- [ ] latent dimensionality;
- [ ] memory use;
- [ ] training stability;
- [ ] convergence speed;
- [ ] inference time;
- [ ] CPU projection performance;
- [ ] Metal projection performance.

### Exit criteria

The repository contains a reproducible comparison showing the effect of the Clifford projection independently of unrelated model changes.

---

# Milestone 4 — Image Autoencoder

## Goal

Learn a compact latent representation of images.

## Initial architecture

```text
Image: 3 × 32 × 32
        ↓
Encoder
        ↓
Latent: C × 8 × 8
        ↓
Decoder
        ↓
Image: 3 × 32 × 32
```

Exact latent channel count will be determined experimentally.

## Tasks

- [ ] Implement `ImageEncoder`.
- [ ] Implement `ImageDecoder`.
- [ ] Add reconstruction loss.
- [ ] Add autoencoder training loop.
- [ ] Add validation split.
- [ ] Save reconstruction samples.
- [ ] Save checkpoints.
- [ ] Resume training.
- [ ] Create `latent_reconstruct`.
- [ ] Measure validation reconstruction loss.
- [ ] Confirm unseen images remain recognisable.

## Exit criteria

The autoencoder reconstructs unseen small validation images with recognisable structure and stable loss.

---

# Milestone 5 — Latent Denoiser

## Goal

Train a model to predict Gaussian noise in the learned image latent space.

## Tasks

- [ ] Convert the maintained epsilon-predictor concepts to spatial latent tensors.
- [ ] Remove NLP-specific artificial vector-to-grid reshaping.
- [ ] Implement sinusoidal timestep embedding.
- [ ] Inject timestep information into denoiser blocks.
- [ ] Implement residual path.
- [ ] Implement output projection matching latent shape.
- [ ] Train against known Gaussian noise.
- [ ] Use MSE noise-prediction objective.
- [ ] Add batch training.
- [ ] Add loss logging.
- [ ] Add seeded training sample generation.

## Exit criteria

Denoising loss decreases on training and validation sets and the model predicts structured noise better than an untrained baseline.

---

# Milestone 6 — End-to-End Latent Diffusion Training

## Goal

Train the encoder/decoder and denoiser as a usable image-generation pipeline.

## Tasks

- [ ] Define whether autoencoder weights are frozen or jointly trained.
- [ ] Encode dataset into latent space.
- [ ] Sample random timesteps.
- [ ] Sample known Gaussian noise.
- [ ] Produce `z_t`.
- [ ] Predict noise.
- [ ] Compute denoising loss.
- [ ] Backpropagate.
- [ ] Update model.
- [ ] Record training metrics.
- [ ] Save model checkpoints.
- [ ] Save optimiser state.
- [ ] Resume deterministically.
- [ ] Create training configuration files.

## Exit criteria

Training can run for multiple epochs, checkpoint, terminate, resume, and continue without state corruption or unexplained loss discontinuity.

---

# Milestone 7 — First Image Generation

## Goal

Generate images from pure Gaussian latent noise.

## Tasks

- [ ] Initialise `z_T ~ N(0,I)`.
- [ ] Implement complete reverse diffusion loop.
- [ ] Use trained denoiser at every reverse step.
- [ ] Decode final `z_0`.
- [ ] Save generated image.
- [ ] Add seed argument.
- [ ] Reproduce outputs by seed.
- [ ] Generate sample grids.
- [ ] Compare intermediate denoising states.
- [ ] Detect NaN/infinite trajectories.

## Primary success criterion

```text
Gaussian noise
    ↓
learned reverse diffusion
    ↓
latent image representation
    ↓
decoder
    ↓
recognisable generated image
```

This is the first major end-to-end project milestone.

---

# Milestone 8 — Generator CLI and Model Packaging

## Goal

Make inference usable independently of training code.

## Tasks

- [ ] Create `latent_generate`.
- [ ] Add checkpoint selection.
- [ ] Add seed.
- [ ] Add output path.
- [ ] Add sample count.
- [ ] Add inference configuration.
- [ ] Add model metadata.
- [ ] Validate checkpoint versions.
- [ ] Add clear runtime errors.
- [ ] Add reproducibility information to generated metadata where practical.

## Example

```bash
./latent_generate   --checkpoint checkpoints/tiny-image.ldc   --seed 42   --count 4   --output outputs/
```

## Exit criteria

A user can build the repository and generate images from a trained checkpoint without invoking the training executable.

---

# Milestone 9 — Experimental Latent Geometry

## Goal

Reintroduce the original research components as controlled experiments.

## Tasks

- [ ] Define `LatentCodec` interface.
- [ ] Establish conventional autoencoder baseline.
- [ ] Implement `CliffordLatentCodec`.
- [ ] Consider wavelet latent transform.
- [ ] Compare dimensionality.
- [ ] Compare reconstruction.
- [ ] Compare denoising loss.
- [ ] Compare training stability.
- [ ] Compare generation quality.
- [ ] Compare memory use.
- [ ] Document results.

## Exit criteria

The effect of each alternative latent geometry can be measured independently of the denoiser and diffusion engine.

---

# Milestone 10 — Improved Denoiser Architecture

## Goal

Move beyond the first compact CNN denoiser after the pipeline is proven.

Candidate directions:

- residual CNN;
- U-Net;
- diffusion transformer;
- state-space denoiser;
- hybrid architecture.

## Tasks

- [ ] Establish benchmark dataset.
- [ ] Establish baseline metrics.
- [ ] Add architecture configuration.
- [ ] Ensure model checkpoint identifies architecture.
- [ ] Benchmark quality versus compute.
- [ ] Retain backward compatibility where practical.

## Exit criteria

At least one larger denoiser improves generation quality over the v1 baseline without breaking reproducibility or checkpointing.

---

# Milestone 11 — Conditioning

## Goal

Move from unconditional generation toward controlled generation.

Conditioning should be added only after unconditional generation works.

Possible order:

1. class conditioning;
2. image conditioning;
3. text conditioning.

## Tasks

- [ ] Introduce conditioning interface.
- [ ] Add null/unconditional path.
- [ ] Implement classifier-free conditioning architecture where appropriate.
- [ ] Add conditional training batches.
- [ ] Add guidance during generation.
- [ ] Test condition adherence.

## Exit criteria

The model generates materially different outputs in response to controlled conditioning signals.

---

# Milestone 12 — Text-to-Image

## Goal

Introduce prompt-conditioned image generation.

## Tasks

- [ ] Select or implement text tokenisation strategy.
- [ ] Implement text encoder.
- [ ] Define text embedding representation.
- [ ] Integrate conditioning into denoiser.
- [ ] Train on paired image-caption data.
- [ ] Add classifier-free guidance.
- [ ] Add prompt CLI.
- [ ] Add negative/unconditional conditioning path if required.

## Exit criteria

Prompt changes produce semantically distinguishable generated images.

---

# Milestone 13 — Metal Acceleration

## Goal

Accelerate validated kernels on Apple Silicon without changing high-level model behaviour.

## Priorities

- convolution;
- linear algebra;
- tensor arithmetic;
- noise generation where appropriate;
- denoising loop.

## Tasks

- [ ] Define backend abstraction.
- [ ] Implement CPU reference backend.
- [ ] Implement Metal buffer ownership.
- [ ] Port tensor operations.
- [ ] Port convolution.
- [ ] Port linear layers.
- [ ] Validate CPU/Metal numerical agreement.
- [ ] Benchmark training.
- [ ] Benchmark generation.
- [ ] Add backend selection.

## Exit criteria

The same checkpoint and configuration can execute using the Metal backend with validated numerical tolerance and measurable performance improvement.

---

# Milestone 14 — CUDA / Additional Backends

## Goal

Extend execution portability after the backend interface has proven itself.

Potential backends:

```text
CUDA
Vulkan compute
CPU SIMD
other accelerator runtimes
```

Backend work must remain independent of model semantics.

---

# Milestone 15 — Scale and Production Hardening

## Goal

Prepare the repository for larger models and external use.

## Tasks

- [ ] dataset streaming;
- [ ] shuffling at scale;
- [ ] data-loader workers;
- [ ] larger image sizes;
- [ ] mixed precision;
- [ ] memory profiling;
- [ ] inference benchmarks;
- [ ] training benchmarks;
- [ ] structured logging;
- [ ] model cards;
- [ ] release artefacts;
- [ ] installation documentation;
- [ ] CI across supported platforms;
- [ ] fuzz/error-path testing;
- [ ] checkpoint migration tools.

---

# Release Targets

## v0.1 — Production Core

Includes:

```text
clean repository
tensor core
deterministic RNG
validated diffusion mathematics
tests
modern CMake
```

## v0.2 — Image Autoencoder

Includes:

```text
image loading
image tensor pipeline
encoder
decoder
reconstruction training
checkpointing
```

## v0.3 — Latent Denoising

Includes:

```text
spatial epsilon predictor
timestep conditioning
diffusion training
noise-prediction loss
```

## v0.4 — First Generator

Includes:

```text
reverse diffusion
latent sampling
image decoding
seeded image generation
generator CLI
```

## v0.5 — Experimental Latent Geometry

Includes:

```text
Clifford latent path
wavelet experiments
baseline comparisons
```

## v0.6 — Conditioning

Includes:

```text
conditioning interface
class/image conditioning
guidance
```

## v0.7 — Text-to-Image

Includes:

```text
text encoder
prompt conditioning
prompt-driven generation
```

## v0.8 — Metal

Includes:

```text
backend interface
Metal compute path
CPU/Metal validation
```

## v1.0 — Production Image Generator

Target characteristics:

```text
reproducible
checkpointed
tested
documented
hardware-aware
image-generating
extensible
```

---

# Immediate Next Work

The next implementation sprint is Milestone 0 followed immediately by Milestone 1.

Priority order:

```text
1. repository cleanup
2. CMake restructuring
3. tensor foundation
4. deterministic RNG
5. unified noise schedule
6. correct q_sample
7. reverse-process coefficients
8. diffusion tests
```

No additional model complexity should be added until those foundations are passing.

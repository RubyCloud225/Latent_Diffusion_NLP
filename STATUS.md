# STATUS.md

# Latent Diffusion Image Generator — Status

**Project status:** Research prototype entering productionisation  
**Current target:** Production Core v0.1  
**End goal:** Native latent-diffusion image generation in C++  
**Primary platform:** CPU reference implementation, with Apple Metal planned  
**Repository:** `RubyCloud225/Latent_Diffusion_NLP`

---

## Current State

The repository currently contains a functioning research codebase for an experimental latent-diffusion NLP pipeline.

The project is not yet an image generator.

The existing implementation demonstrates several important components required by the future image pipeline, but the architecture is still organised around tokenised text and contains legacy, experimental, and partially integrated model paths.

Productionisation has now begun.

---

# Implemented

## Build

- [x] CMake build exists.
- [x] C++17 configuration exists.
- [x] Main executable can be compiled.
- [x] Shell build/run helper exists.

## NLP preprocessing

- [x] Basic BPE implementation.
- [x] Tokenisation.
- [x] Deterministic embedding initialisation.
- [x] 64-dimensional embedding representation.
- [x] Experimental Clifford compression.
- [x] FP16 conversion utilities.
- [x] Huffman serialization experiment.

These components are considered research/legacy for the image-generation branch and will not define the production image input pipeline.

## Diffusion research

- [x] Beta scheduling experiments.
- [x] Gaussian noise injection.
- [x] Reverse-process experiments.
- [x] Diffusion model abstraction experiments.
- [x] NLL-related probability utilities.
- [x] Clamping/stability experiments.
- [x] Functional epsilon-predictor hook experiments.

## Neural-network research

- [x] Custom convolutional layer implementation.
- [x] ReLU.
- [x] Pooling.
- [x] Flatten.
- [x] Fully connected layer.
- [x] Persistent epsilon-predictor architecture.
- [x] Sinusoidal timestep embedding.
- [x] Custom Adam optimiser.
- [x] Partial checkpoint support.

## Known corrections already made in newer model code

The maintained neural-network code records fixes for several earlier bugs:

- [x] convolution weights no longer initialise identically by accident;
- [x] excessive inner-loop debug output removed;
- [x] pooling dimension ordering corrected;
- [x] pooling allocation behaviour corrected;
- [x] fully connected layer now applies trainable weights;
- [x] network execution respects layer ordering;
- [x] epsilon output no longer applies ReLU;
- [x] epsilon predictor persists model state between calls;
- [x] timestep embeddings are included.

---

# Not Yet Production Ready

## Repository structure

- [ ] Build artefacts are still committed under `build/`.
- [ ] Legacy NLP code is mixed with maintained code.
- [ ] Source and public headers are not cleanly separated.
- [ ] Test structure is not established.
- [ ] Training and inference applications are not separated.
- [ ] Production configuration system does not exist.

## Build integration

The current CMake target does not include all maintained implementation files.

Notably, the primary target does not currently compile:

```text
ClassicalDiT/NN/EpsilonPredictor.cpp
ClassicalDiT/NN/NeuralNetworkLayers.cpp
ClassicalDiT/training.cpp
```

This must be corrected during Milestone 0.

## Diffusion mathematics

The codebase currently contains overlapping diffusion formulations.

The newer epsilon-predictor documentation describes the expected cumulative-alpha formulation:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

However the current `GaussianDiffusion::forward()` implementation performs incremental additive Gaussian noise.

These must be consolidated into one production definition.

## Reverse diffusion

The current `GaussianDiffusion::reverse()` still estimates epsilon using newly sampled random noise.

This is placeholder behaviour and cannot form the production generation path.

Required production behaviour:

```text
epsilon = denoiser(x_t, t)
```

## Training

The repository has training experiments but does not yet provide a complete end-to-end gradient path through all denoiser parameters.

The existing Adam implementation updates a parameter vector supplied to `GaussianDiffusion::train`, but this does not yet constitute full network backpropagation through the maintained epsilon-predictor.

## Checkpointing

The epsilon-predictor checkpoint currently saves only fully connected weights and biases.

Convolution parameters are not fully persisted.

Production checkpointing must save all trainable state.

## Image support

- [ ] No production image tensor type.
- [ ] No image dataset pipeline.
- [ ] No production image encoder.
- [ ] No production image decoder.
- [ ] No image autoencoder training.
- [ ] No spatial latent representation.
- [ ] No image-generation executable.
- [ ] No generated-image checkpoint.

---

# Historical Training Result

The repository contains a December 2025 training report showing divergence in an earlier version of the system.

Reported behaviour included:

```text
MSE ≈ 1.16 × 10^15
NLL ≈ 1.16 × 10^15
large unstable predictions
```

The report correctly identified likely causes including:

- insufficient normalisation;
- incomplete parameter updates;
- metric bugs;
- lack of gradient clipping;
- unstable training behaviour.

Several neural-network implementation issues have since been addressed, but the new production path will establish a clean tested baseline rather than assuming those earlier experiments are representative of the maintained architecture.

---

# Current Productionisation Phase

## Milestone 0 — Repository Rehabilitation

**Status:** In progress

### Documentation

- [x] Production architecture defined.
- [x] Roadmap defined.
- [x] Current state documented.
- [x] Changelog baseline defined.

### Codebase cleanup

- [x] Remove tracked build outputs.
- [x] Expand `.gitignore`.
- [x] Introduce production directory layout.
- [x] Move NLP-specific implementation to legacy namespace/directory.
- [ ] Move experimental Clifford implementation behind an explicit interface.
- [ ] Replace monolithic `main.cpp` responsibilities.

### Build

- [ ] Rewrite root CMake configuration.
- [ ] Add core library target.
- [ ] Add application targets.
- [ ] Add test target.
- [ ] Compile maintained epsilon predictor.
- [ ] Compile maintained neural-network layers.
- [ ] Add warnings.

---

# Next Milestone

## Milestone 1 — Production Diffusion Core

Planned first implementation components:

```text
Tensor
RandomGenerator
NoiseSchedule
GaussianDiffusion
Sampler
```

First mathematical implementation target:

```text
beta_t
alpha_t
alpha_bar_t
q_sample(z0, t, epsilon)
```

First validation target:

Given identical:

```text
input
timestep
noise
seed
configuration
```

the engine must produce identical output across repeated CPU executions.

---

# Image Generation Progress

Current progress toward end-to-end image generation:

```text
Repository foundation       ███░░░░░░░
Diffusion mathematics       ████░░░░░░
Trainable neural layers     ████░░░░░░
Image pipeline              ░░░░░░░░░░
Image autoencoder           ░░░░░░░░░░
Spatial denoiser            ░░░░░░░░░░
Latent diffusion training   ░░░░░░░░░░
Reverse generation          ░░░░░░░░░░
Conditioning                ░░░░░░░░░░
Hardware acceleration       ░░░░░░░░░░
```

The bar is qualitative and intended only as a development overview.

---

# Current Architectural Decisions

## Retain

- C++ implementation.
- Custom diffusion mathematics.
- Custom optimiser work.
- Custom neural layers where practical.
- timestep embeddings.
- deterministic design.
- Clifford latent research.
- zero/minimal dependency philosophy.

## Refactor

- monolithic `main.cpp`;
- nested vectors for image tensors;
- overlapping diffusion classes;
- training architecture;
- checkpoint format;
- CMake structure;
- random-number ownership.

## Replace for image path

- BPE input pipeline;
- text embedding table;
- NLP-specific vector latent assumptions;
- artificial 1D latent-to-square-grid conversion.

## Preserve as experiment

- Clifford compression;
- adaptive beta scheduling;
- Huffman latent serialization;
- alternative probabilistic objectives.

## Clifford Projection

The original `clifford_compress()` research implementation does not perform a mathematical Clifford projection.

It currently:

```text
takes the first eight source values
converts those values to FP16
retains a complete copy of the original vector
```

This behaviour is now classified as legacy prototype behaviour.

The production implementation will replace it with:

\[
c = Px
\]

where:

```text
x ∈ R^64
P ∈ R^(8×64)
c ∈ R^8
```

and the eight output coefficients form a `Cl(3,0)` multivector.

### Production decision

- [x] Use a trainable `64 → 8` projection matrix.
- [x] Treat projection weights as model parameters.
- [x] Keep projection differentiable.
- [x] Support gradient flow into the source latent.
- [x] Store actual compressed coefficients separately from the source latent.
- [x] Maintain FP32 coefficients.
- [x] Support derived FP16 coefficients.
- [x] Use CPU implementation as the reference.
- [x] Target Apple Metal as the first GPU implementation.

### Test decision

Production weights will be trainable, but tests will use a deterministic fixed orthonormal projection.

Known-answer test:

```text
input:
[1, 2, 3, 4, 5, 6, 7, 8, ..., 64]

expected:
scalar = 1
e1     = 2
e2     = 3
e3     = 4
e12    = 5
e13    = 6
e23    = 7
e123   = 8
```

This same fixture will validate:

```text
CPU projection
FP16 conversion
Metal projection
CPU ↔ Metal parity
```

### Current implementation state

- [ ] `FP16.hpp`
- [ ] `CliffordMultivector.hpp`
- [ ] `CliffordProjection.hpp`
- [ ] CPU projection implementation
- [ ] backward projection implementation
- [ ] known-answer tests
- [ ] FP16 tests
- [ ] Metal projection
- [ ] CPU/Metal parity tests

---

# Immediate Task Queue

1. Clean repository layout.
2. Remove `build/` from version control.
3. Modernise CMake.
4. Create core library target.
5. Create tests target.
6. Add contiguous `Tensor`.
7. Add seeded `RandomGenerator`.
8. Consolidate diffusion schedule.
9. Implement `q_sample`.
10. Add diffusion validation tests.
11. Introduce image tensor and image I/O.
12. Begin autoencoder implementation.

---

# Definition of "Production Ready"

For this project, production ready does not mean merely compiling.

A production component must have:

- clear ownership and interface;
- deterministic behaviour where expected;
- validation of invalid inputs;
- automated tests;
- checkpoint/version compatibility where relevant;
- no hidden placeholder behaviour;
- documented configuration;
- no reliance on generated repository artefacts;
- reproducible execution.

The repository as a whole reaches the first production image-generator milestone only when a clean build can train, checkpoint, reload, and generate a reproducible image from latent Gaussian noise.

# CHANGELOG.md

# Changelog

All notable changes to the production image-generation line of this project will be documented in this file.

The project began as an experimental latent-diffusion NLP research implementation. The changelog below establishes a clean baseline for the transition into a production-oriented image generator.

Format is based broadly on Keep a Changelog principles, while version numbers reflect the internal development roadmap.

---

## [Unreleased]

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

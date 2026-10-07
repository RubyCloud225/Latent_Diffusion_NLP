# DESIGN.md

# Latent Diffusion Image Generator — Design

## 1. Purpose

This repository is being evolved from an experimental C++ latent-diffusion NLP implementation into a production-oriented latent diffusion engine with the end goal of generating images.

The project will preserve the original close-to-metal design philosophy:

- C++ as the primary implementation language.
- Minimal external dependencies.
- Explicit control over diffusion mathematics, model execution, memory layout, and checkpointing.
- CPU-first correctness before hardware acceleration.
- Modular support for alternative latent transforms, denoisers, schedules, and compute backends.
- No dependency on an existing end-to-end diffusion framework such as Stable Diffusion.

The first production objective is not high-resolution text-to-image generation. It is to demonstrate a correct, trainable, reproducible image latent-diffusion pipeline capable of generating recognisable small images from Gaussian noise.

---

## 2. Design Goals

### 2.1 Primary goals

1. Implement a mathematically consistent latent diffusion core.
2. Train an image encoder and decoder capable of reconstructing input images.
3. Train a denoiser to predict noise in the learned image latent space.
4. Generate images from pure latent noise.
5. Support deterministic training and generation through explicit random seeds.
6. Support checkpoint save, load, resume, and inference.
7. Maintain a clean separation between research components and production components.
8. Preserve custom latent representations such as Clifford compression as optional experimental modules.
9. Provide a path to Apple Metal and later CUDA or other accelerator backends.
10. Keep the core implementation inspectable and testable without hiding model behaviour behind a large framework.

### 2.2 Non-goals for the initial production release

The first production milestones do not require:

- 512×512 or 1024×1024 generation.
- text-to-image conditioning;
- CLIP compatibility;
- Stable Diffusion checkpoint compatibility;
- distributed training;
- multi-GPU execution;
- web APIs;
- production model hosting;
- photorealistic image quality.

These may be introduced only after the base generation pipeline is validated.

---

## 3. Current Research Lineage

The repository currently contains an experimental latent diffusion NLP pipeline built around:

- custom BPE tokenisation;
- deterministic dense embeddings;
- Clifford-inspired latent compression;
- Gaussian diffusion;
- adaptive beta scheduling;
- a custom Adam optimiser;
- custom convolutional and fully connected layers;
- timestep embeddings;
- an epsilon-prediction network;
- custom serialization and checkpoint experiments.

The existing implementation contains valuable research components but also multiple overlapping diffusion paths, legacy neural-network layers, NLP-specific code embedded inside the executable, and incomplete training integration.

Productionisation therefore requires architectural separation rather than incremental expansion of the existing `main.cpp`.

---

## 4. Target System Architecture

The target image-generation path is:

```text
Input Image
    │
    ▼
Image Preprocessing
    │
    ▼
Image Encoder
    │
    ▼
Latent z₀
    │
    ├──────────────► Forward Diffusion q(zₜ | z₀)
    │                         │
    │                         ▼
    │                  Noisy Latent zₜ
    │                         │
    │              timestep ──┤
    │                         ▼
    │                 Denoiser εθ(zₜ, t)
    │                         │
    │                         ▼
    │                Predicted Noise ε̂
    │                         │
    ▼                         ▼
Training Loss ◄──────── Known Noise ε

Generation:

zT ~ N(0, I)
    │
    ▼
Reverse Diffusion
    │
    ▼
z₀
    │
    ▼
Image Decoder
    │
    ▼
Generated Image
```

The image pipeline will be divided into six major systems:

1. Core tensor and numerical utilities.
2. Image input/output and preprocessing.
3. Latent encoder/decoder.
4. Diffusion mathematics and sampling.
5. Denoising network.
6. Training, checkpointing, and inference applications.

---

## 5. Repository Architecture

Target structure:

```text
Latent_Diffusion_NLP/
│
├── CMakeLists.txt
├── README.md
├── DESIGN.md
├── ROADMAP.md
├── STATUS.md
├── CHANGELOG.md
├── LICENSE
│
├── include/
│   └── latent/
│       ├── core/
│       │   ├── Tensor.hpp
│       │   ├── Random.hpp
│       │   └── Types.hpp
│       │
│       ├── diffusion/
│       │   ├── NoiseSchedule.hpp
│       │   ├── GaussianDiffusion.hpp
│       │   └── Sampler.hpp
│       │
│       ├── nn/
│       │   ├── Conv2D.hpp
│       │   ├── Linear.hpp
│       │   ├── Activation.hpp
│       │   ├── Loss.hpp
│       │   └── Optimizer.hpp
│       │
│       ├── model/
│       │   ├── TimestepEmbedding.hpp
│       │   ├── Denoiser.hpp
│       │   ├── ImageEncoder.hpp
│       │   └── ImageDecoder.hpp
│       │
│       ├── image/
│       │   ├── Image.hpp
│       │   └── ImageIO.hpp
│       │
│       ├── checkpoint/
│       │   └── Checkpoint.hpp
│       │
│       └── experimental/
│           ├── CliffordLatent.hpp
│           └── WaveletLatent.hpp
│
├── src/
│   ├── core/
│   ├── diffusion/
│   ├── nn/
│   ├── model/
│   ├── image/
│   ├── checkpoint/
│   └── experimental/
│
├── apps/
│   ├── train.cpp
│   ├── generate.cpp
│   ├── reconstruct.cpp
│   └── inspect_checkpoint.cpp
│
├── tests/
│   ├── core/
│   ├── diffusion/
│   ├── nn/
│   ├── model/
│   └── integration/
│
├── configs/
│   ├── tiny_image.toml
│   └── base_image.toml
│
├── scripts/
│
├── checkpoints/
│   └── .gitkeep
│
└── legacy/
    └── nlp/
```

Generated build products must not be committed to source control.

---

## 6. Core Tensor Representation

The existing code frequently represents data with nested `std::vector` structures. That is useful for experimentation but becomes difficult to validate and optimise for image workloads.

Production code should introduce a contiguous tensor abstraction.

Initial shape support:

```text
[D]
[N, D]
[C, H, W]
[N, C, H, W]
```

Conceptual interface:

```cpp
class Tensor {
public:
    Tensor(std::vector<size_t> shape);

    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    const std::vector<size_t>& shape() const;
    size_t size() const;

    double* data();
    const double* data() const;
};
```

The first implementation may use `double` for numerical validation. A later precision policy should support:

- FP64 for numerical tests;
- FP32 for training;
- FP16/BF16 where backend support permits.

The tensor implementation must keep contiguous storage so SIMD, Metal, CUDA, and other backends can be introduced without redesigning every model layer.

---

## 7. Image Representation

Images are represented internally as tensors with shape:

```text
[C, H, W]
```

where:

```text
C = 1 for greyscale
C = 3 for RGB
```

Initial preprocessing:

1. Load image.
2. Convert to expected channel layout.
3. Resize or reject incompatible dimensions according to configuration.
4. Convert integer colour values to floating point.
5. Normalise to `[-1, 1]`.
6. Feed tensor into the encoder.

Decoder output is clamped to `[-1, 1]` and transformed back into an image representation before writing.

The initial image writer may use a simple format such as PPM while the mathematical pipeline is being validated. PNG support can follow through an isolated image-I/O dependency or self-contained implementation.

---

## 8. Image Autoencoder

Latent diffusion requires a learned mapping:

\[
E(x) = z
\]

and:

\[
D(z) = \hat{x}
\]

where:

- `x` is an image;
- `z` is its compressed latent representation;
- `E` is the encoder;
- `D` is the decoder.

The first encoder should use a small convolutional hierarchy:

```text
RGB Image
  ↓
Conv
  ↓
Activation
  ↓
Downsample
  ↓
Conv
  ↓
Activation
  ↓
Downsample
  ↓
Latent Tensor
```

The decoder reverses the process using upsampling and convolution or transposed convolution.

The first autoencoder milestone is reconstruction, not generation.

Success condition:

```text
input image → encoder → latent → decoder → recognisable reconstruction
```

The autoencoder must be validated independently before diffusion training begins.

---

## 9. Latent Interface

Diffusion must not depend directly on one latent encoding strategy.

Conceptual interface:

```cpp
class LatentCodec {
public:
    virtual ~LatentCodec() = default;

    virtual Tensor encode(const Tensor& image) = 0;
    virtual Tensor decode(const Tensor& latent) = 0;
};
```

Initial implementation:

```text
ImageAutoencoderCodec
```

Experimental implementations may later include:

```text
CliffordLatentCodec
WaveletLatentCodec
HybridLatentCodec
```

This allows the original Clifford work to be evaluated against a conventional learned latent baseline rather than being inseparably embedded in the pipeline.

---

## 10. Diffusion Mathematics

The production diffusion engine will use one consistent definition of the forward process.

For timestep `t`:

\[
\alpha_t = 1 - \beta_t
\]

and cumulative product:

\[
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
\]

Forward noising:

\[
z_t =
\sqrt{\bar{\alpha}_t}z_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
\]

where:

\[
\epsilon \sim \mathcal{N}(0,I)
\]

The forward API should accept explicit noise:

```cpp
Tensor q_sample(
    const Tensor& z0,
    int timestep,
    const Tensor& noise
);
```

Explicit noise is important for:

- deterministic tests;
- reproducibility;
- direct comparison between predicted and true noise;
- debugging.

---

## 11. Noise Schedule

Noise scheduling becomes a dedicated component.

Conceptual interface:

```cpp
class NoiseSchedule {
public:
    explicit NoiseSchedule(const ScheduleConfig& config);

    double beta(int t) const;
    double alpha(int t) const;
    double alpha_bar(int t) const;

    double sqrt_alpha_bar(int t) const;
    double sqrt_one_minus_alpha_bar(int t) const;
};
```

Initial schedules:

- linear;
- cosine, after the linear implementation is validated.

All required cumulative values should be precomputed.

The existing adaptive beta research may later be reintroduced as an experimental schedule once the standard baseline is correct.

---

## 12. Denoiser

The denoiser learns:

\[
\epsilon_\theta(z_t,t)
\]

The initial production denoiser should remain deliberately small.

Inputs:

```text
noisy latent zₜ
timestep t
```

Output:

```text
predicted noise ε̂
```

The first implementation can evolve the existing `EpsilonPredictor`, but it must operate directly on spatial latent tensors rather than flattening an NLP embedding into an artificial square grid.

Target architecture:

```text
zₜ
 │
 ├── Conv block
 │
 ├── timestep conditioning
 │
 ├── Conv block
 │
 ├── residual path
 │
 └── output projection
       ↓
      ε̂
```

A larger U-Net or Diffusion Transformer may be introduced only after the small denoiser proves the training and sampling pipeline.

---

## 13. Training Objective

The initial denoising objective is noise-prediction MSE:

\[
L =
\mathbb{E}_{z_0,t,\epsilon}
\left[
\|\epsilon - \epsilon_\theta(z_t,t)\|^2
\right]
\]

Training iteration:

```text
image
 ↓
encoder
 ↓
z₀
 ↓
sample timestep t
 ↓
sample ε ~ N(0,I)
 ↓
q_sample(z₀, t, ε)
 ↓
zₜ
 ↓
denoiser(zₜ, t)
 ↓
ε̂
 ↓
MSE(ε̂, ε)
 ↓
backpropagation
 ↓
optimizer update
```

The existing NLL experiments remain valuable research history but should not be the primary v0.1 image denoising objective.

---

## 14. Gradient System

A production trainable network requires gradients to propagate through all trainable layers.

The existing code must be audited for:

- forward implementations;
- gradient calculation;
- parameter ownership;
- gradient accumulation;
- zeroing;
- optimiser integration;
- shape validation.

Every trainable layer should expose a consistent interface.

Conceptually:

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

    virtual std::vector<ParameterRef> parameters() = 0;
};
```

This should remain small and explicit rather than becoming a general-purpose deep-learning framework.

---

## 15. Optimisation

The existing custom Adam implementation should be retained and migrated behind a clean optimiser interface.

Requirements:

- one optimiser state per trainable parameter;
- bias correction;
- configurable learning rate;
- configurable beta values;
- epsilon;
- gradient clipping support;
- checkpointed optimiser state;
- deterministic resumption.

Initial optimiser:

```text
Adam
```

Potential later additions:

```text
AdamW
SGD
```

---

## 16. Reverse Diffusion and Sampling

Generation starts with:

\[
z_T \sim \mathcal{N}(0,I)
\]

and iteratively computes:

```text
zT → zT-1 → ... → z1 → z0
```

The production sampler must use the trained denoiser.

No random placeholder epsilon estimates are permitted in the generation path.

Conceptual interface:

```cpp
Tensor sample_step(
    const Tensor& z_t,
    int timestep,
    Denoiser& model,
    RandomGenerator& rng
);
```

Full generation:

```cpp
Tensor sample(
    const std::vector<size_t>& latent_shape,
    Denoiser& model,
    uint64_t seed
);
```

A deterministic seed should reproduce the same image for the same model checkpoint and configuration.

---

## 17. Checkpoint Format

A production checkpoint must contain enough state to reproduce inference and resume training.

Required fields:

```text
format version
model architecture version
training step
epoch
configuration
encoder parameters
decoder parameters
denoiser parameters
optimizer state
noise schedule configuration
random seed/state where appropriate
```

Checkpoint files should include explicit versioning.

Example:

```text
LDCP
version: 1
```

The current epsilon-predictor checkpoint only stores the fully connected layer. That is insufficient for production and will be replaced.

---

## 18. Applications

The project should build separate executables.

### `latent_train`

```bash
./latent_train --config configs/tiny_image.toml
```

Responsibilities:

- load dataset;
- initialise or resume model;
- train;
- validate;
- log metrics;
- save checkpoints.

### `latent_reconstruct`

```bash
./latent_reconstruct   --checkpoint checkpoints/autoencoder.ldc   --input example.ppm   --output reconstructed.ppm
```

Used to validate the autoencoder.

### `latent_generate`

```bash
./latent_generate   --checkpoint checkpoints/model.ldc   --seed 42   --output generated.ppm
```

This becomes the primary end-user inference path.

---

## 19. Configuration

Hard-coded training constants should move into explicit configuration.

Example:

```toml
[image]
width = 32
height = 32
channels = 3

[latent]
channels = 4
width = 8
height = 8

[diffusion]
timesteps = 1000
schedule = "linear"
beta_start = 0.0001
beta_end = 0.02

[training]
batch_size = 16
epochs = 100
learning_rate = 0.0001
seed = 42
```

Configuration loading should validate invalid or incompatible values at startup.

---

## 20. Testing Strategy

Testing is part of the architecture, not an afterthought.

### Core tests

- tensor shape;
- indexing;
- contiguous storage;
- bounds behaviour;
- deterministic RNG.

### Diffusion tests

- beta range;
- alpha calculation;
- cumulative alpha;
- timestep boundaries;
- `q_sample` at `t = 0`;
- high-noise timestep behaviour;
- deterministic noise;
- no NaN or infinity.

### Neural-network tests

- convolution output shape;
- linear layer output shape;
- activation behaviour;
- numerical gradient checks;
- optimiser update;
- checkpoint round trip.

### Autoencoder tests

- encoder output dimensions;
- decoder output dimensions;
- reconstruction path;
- training loss reduction on tiny synthetic data.

### Integration tests

- train a tiny model;
- save checkpoint;
- reload checkpoint;
- reproduce output;
- generate from seeded Gaussian noise.

---

## 21. Determinism

All stochastic operations must derive from an explicit RNG object or seed.

Avoid constructing unseeded local random engines inside diffusion functions.

Required:

```cpp
RandomGenerator rng(seed);
```

This controls:

- dataset shuffle;
- initialisation;
- timestep sampling;
- Gaussian noise;
- generation.

Reproducibility is required for debugging and scientific comparison.

---

## 22. Logging and Metrics

Training should report at minimum:

```text
epoch
global step
learning rate
denoising loss
reconstruction loss
gradient norm
samples/second
checkpoint path
```

The image pipeline should periodically save visual reconstruction and generation samples.

Research metrics can be introduced after the first valid generator exists.

---

## 23. Performance Strategy

Optimisation order:

1. mathematical correctness;
2. deterministic behaviour;
3. clean profiling;
4. contiguous memory;
5. compiler optimisation;
6. SIMD;
7. multithreading;
8. Metal backend;
9. additional accelerator backends.

Performance changes must not alter reference numerical behaviour without explicit tests.

---

## 24. Backend Strategy

Initial backend:

```text
CPU / portable C++17 or newer
```

Planned backend interface:

```text
CPUBackend
MetalBackend
CUDABackend
```

The tensor and model APIs should avoid assumptions that prevent GPU execution.

Apple Metal is the first planned accelerated backend because development is primarily performed on Apple Silicon.

---

## 25. Legacy Code Policy

Existing research code will not be deleted merely because it is no longer part of the production path.

It should be moved under:

```text
legacy/nlp/
```

or:

```text
experimental/
```

as appropriate.

Legacy code must not be linked into production binaries unless explicitly required.

Deprecated layer implementations should be retained only until the production replacements are validated.

---

## 26. Experimental Clifford Latent Path

The Clifford representation remains part of the research direction, but it must be benchmarked against a baseline.

The architecture should permit:

```text
Image
 ↓
Encoder
 ↓
standard latent
```

versus:

```text
Image
 ↓
Encoder
 ↓
Clifford transform
 ↓
compressed geometric latent
```

Comparison criteria may include:

- reconstruction quality;
- denoising loss;
- generation quality;
- latent dimensionality;
- storage;
- training time;
- sampling time;
- numerical stability.

This separation makes the effect of Clifford compression measurable.

---


## Clifford Latent Projection

The production image model will use a **trainable projection from the source latent representation into an eight-component Clifford multivector**.

The projection is not implemented as truncation of the first eight latent values.

For a source latent vector

\[
x \in \mathbb{R}^{64}
\]

the Clifford projection layer contains a trainable matrix

\[
P \in \mathbb{R}^{8 \times 64}
\]

and computes

\[
c = Px
\]

where

\[
c \in \mathbb{R}^{8}.
\]

The eight output coefficients are interpreted as the basis coefficients of \(Cl(3,0)\):

\[
M =
c_0
+ c_1e_1
+ c_2e_2
+ c_3e_3
+ c_4e_{12}
+ c_5e_{13}
+ c_6e_{23}
+ c_7e_{123}.
\]

The coefficient ordering is fixed:

```text
0   scalar
1   e1
2   e2
3   e3
4   e12
5   e13
6   e23
7   e123
```

### Production behaviour

In the production training path, the projection matrix is a normal trainable model parameter.

The forward operation is:

\[
c_i =
\sum_{j=0}^{63}
P_{ij}x_j.
\]

Gradients propagate through both the projection weights and the input latent:

\[
\frac{\partial L}{\partial P}
=
\frac{\partial L}{\partial c}x^T
\]

and

\[
\frac{\partial L}{\partial x}
=
P^T
\frac{\partial L}{\partial c}.
\]

This allows the latent projection to learn which combinations of source latent dimensions should populate the Clifford basis blades.

### Validation behaviour

Tests must not rely on randomly initialised or trained projection weights.

The test suite will instead use a deterministic fixed orthonormal projection matrix.

For the initial known-answer test, the first eight basis vectors of \(\mathbb{R}^{64}\) are used as the rows of the projection:

```text
P[0,0] = 1
P[1,1] = 1
P[2,2] = 1
...
P[7,7] = 1
```

with all remaining coefficients equal to zero.

Therefore:

\[
PP^T = I_8.
\]

Given the test input:

```text
x = [1, 2, 3, 4, 5, 6, 7, 8, ..., 64]
```

the expected Clifford coefficients are exactly:

```text
scalar = 1
e1     = 2
e2     = 3
e3     = 4
e12    = 5
e13    = 6
e23    = 7
e123   = 8
```

This becomes the authoritative known-answer test for both CPU and GPU implementations.

---

## Clifford Component Separation

The Clifford implementation is divided into separate responsibilities.

```text
CliffordMultivector
        │
        └── represents the eight Cl(3,0) coefficients

CliffordProjection
        │
        ├── trainable 64 → 8 projection
        ├── forward pass
        ├── backward pass
        └── projection parameters

CliffordAlgebra
        │
        ├── geometric product
        ├── inner product
        ├── outer product
        ├── reverse
        └── norm
```

The projection operation and the geometric algebra operations must not be conflated.

`CliffordProjection` determines how the source latent is mapped into a multivector.

`CliffordAlgebra` operates on multivectors once they have been constructed.

---

## Clifford Source Layout

Relevant production files:

```text
include/latent/experimental/
├── CliffordMultivector.hpp
├── CliffordProjection.hpp
└── CliffordAlgebra.hpp

src/experimental/
├── CliffordProjection.cpp
└── CliffordAlgebra.cpp
```

FP16 conversion utilities belong in the core numerical layer:

```text
include/latent/core/FP16.hpp
```

The CPU implementation is the reference implementation.

GPU implementations must reproduce CPU results within documented numerical tolerances.

---

## FP16 Representation

Each Clifford multivector may expose both:

```text
FP32 coefficients
FP16 coefficient representation
```

The FP16 representation is derived from the projected Clifford coefficients, not from the uncompressed source latent.

Conceptually:

```text
64D source latent
       │
       ▼
trainable projection
       │
       ▼
8D Clifford FP32 coefficients
       │
       └────────► FP16 conversion
```

The compressed object does not retain a full copy of the original 64-dimensional vector.

The original latent remains owned by the caller or tensor graph.

---

## Metal Clifford Projection

The Metal implementation performs the same operation as the CPU reference:

\[
c = Px.
\]

Each GPU thread is responsible for one output Clifford coefficient:

```text
thread 0 → scalar
thread 1 → e1
thread 2 → e2
thread 3 → e3
thread 4 → e12
thread 5 → e13
thread 6 → e23
thread 7 → e123
```

Each thread computes:

\[
c_i =
\sum_{j=0}^{63}
P_{ij}x_j.
\]

Metal therefore accelerates the actual projection rather than merely copying or truncating source values.

Relevant files:

```text
src/backend/metal/
├── CliffordProjection.metal
└── CliffordProjectionMetal.mm
```

The initial Metal implementation may copy results back to CPU memory for validation.

The production training implementation should subsequently retain tensors in GPU memory and pass the projected latent directly into downstream GPU operations.

---

## CPU/GPU Numerical Contract

The CPU implementation defines reference behaviour.

The Metal implementation must satisfy:

```text
same input
same projection matrix
        ↓
CPU Clifford coefficients
        ≈
Metal Clifford coefficients
```

FP32 comparison tolerance:

```text
1e-5
```

or tighter where supported.

FP16 comparison uses a looser tolerance appropriate to binary16 precision.

No GPU implementation is considered valid until it passes the same known-answer projection test as the CPU implementation.

---

## 27. Definition of the First Successful Generator

The first successful image-generation milestone is achieved when the repository can:

1. build from a clean checkout;
2. train a small image autoencoder;
3. reconstruct unseen validation images;
4. train a denoiser in latent space;
5. begin generation from Gaussian latent noise;
6. execute a complete reverse diffusion chain;
7. decode the final latent;
8. save a recognisable generated image;
9. reproduce the same image from the same checkpoint and seed.

This defines the first end-to-end production proof.

---

## 28. Long-Term Direction

Once the baseline generator is validated, the architecture can expand toward:

```text
larger image resolutions
better autoencoders
U-Net or DiT denoisers
classifier-free guidance
text conditioning
multimodal conditioning
latent geometry experiments
flow matching
alternative stochastic processes
Metal acceleration
CUDA acceleration
distributed training
model serving
```

The project should remain modular enough that those additions extend the engine rather than require another rewrite.

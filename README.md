# Latent Diffusion NLP: Clifford-Optimized Generative Modeling (C++)

A high-performance latent diffusion pipeline for natural language processing, combining tokenization, geometric embedding compression, and CNN-based noise prediction. This project demonstrates an end-to-end approach to generative modeling optimized for hardware efficiency.

---
## Zero Dependency Diffusion

This project implements a complete generative pipeline in pure C++17, treating natural language as a continuous geometric field rather than discrete tokens. By bypassing standard deep learning libraries, the engine achieves a "close-to-metal" implementation of stochastic differential equations (SDEs).

---

## Execution Logic: `Main.cpp`

The core framework is orchestrated within `Main.cpp`, which manages the lifecycle of latent representations through four distinct phases:

### 1. Tokenization & Embedding
The pipeline processes raw text into a compact vocabulary using a custom **BPE (Byte-Pair Encoding) Tokenizer**. 
- **Dense Embedding:** Tokens are mapped into a 64-dimensional latent space.
- **Clifford Compression:** Embeddings are projected into a Clifford Manifold $C\ell_{p,q}(\mathbb{R})$ to retain multilinear relationships. This geometric approach allows for radical parameter reduction while maintaining the physical integrity of the token relationships.

### 2. Gaussian Diffusion Process
The forward diffusion process gradually injects Gaussian noise into the latent embeddings according to a variance schedule $\beta_t$:

$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3. CNN $\epsilon$-Prediction Network
A **Convolutional Neural Network (CNN)** is utilized as the core epsilon-predictor. Unlike standard MLPs, the CNN captures local spatial dependencies within the latent sequence:
- **Objective:** Minimize the Negative Log-Likelihood (NLL) of the noise distribution.
- **Optimization:** Gradients are computed via the `NormalDist` module to update the CNN kernels.

$$\mathcal{L} = -\log p_\theta(y \mid \mu, \sigma) = \frac{1}{2} \left( \frac{y - \mu}{\sigma} \right)^2 + \log(\sigma) + \frac{1}{2} \log(2\pi)$$

### 4. Adaptive BetaSchedule & Training Loop
The training loop utilizes a dynamic **BetaSchedule** that anneals the noise variance based on epoch progress and loss statistics. This ensures the model balances coarse-grained noise injection with fine-grained reconstruction:

$$\beta_{epoch} = \beta_0 \left(1 - \frac{epoch}{T}\right) + \frac{epoch}{T}$$

The reverse diffusion estimate $\mu_{t-1}$ is then used to reconstruct clean latent states from noisy inputs:

$$\mu_{t-1} = \frac{x_t - \beta_t \epsilon_\theta(x_t, t)}{\sqrt{1 - \beta_t}}$$

---

## Architecture Flow
### 1. High-Performance Tokenization & Geometric Embedding
The pipeline initiates with a custom-engineered preprocessing layer designed for hardware-aware efficiency:

- **Adaptive BPE Tokenizer:** A from-scratch Byte-Pair Encoding implementation that optimizes vocabulary entropy. Unlike standard libraries, this is designed for minimal overhead during the C++ inference lifecycle.
- **Clifford Algebra-Based Compression ($C\ell_{p,q}$):** Token embeddings are projected into a Clifford Manifold. By utilizing the geometric product:
  $$uv = u \cdot v + u \wedge v$$
  we preserve the multilinear and rotational relationships between tokens. This allows for a **geometric representation** that retains high-dimensional semantic integrity at a significantly reduced parameter count compared to traditional Euclidean embeddings.
- **Deterministic Hashing:** Dense vector initialization uses a deterministic hashing scheme to ensure consistent latent mapping across distributed silicon clusters.

```mermaid
graph LR
    A[Raw Text] --> B[BPE Tokenizer]
    B --> C[Clifford Manifold]
    C --> D[Huffman Serialization]
    D --> E[Forward SDE / Noise]
    E --> F[Epsilon Predictor Training]
    F --> G[Reverse Diffusion Sampling]
    G --> H[Reconstructed Latents]
```

```mermaid
graph LR
    X_prev[x t-1] --> |Add noise| X_t[x t]
    Beta_t[Noise Variance Beta_t] --> X_t
    Noise[Epsilon ~ N,0,I] --> X_t
    X_t --> |Predict Eps_theta| PredictedEps[Predicted Noise]
    PredictedEps --> |Reverse mean Mu t-1| X_recon[x t-1 reconstructed]
````
```mermaid
graph LR
    Epoch[Epoch] --> Beta_t[Beta_epoch]
    BetaStart[Beta_0] --> Beta_t
    BetaEnd[1.0] --> Beta_t
    Losses[NLL / Entropy Loss] --> Beta_t
    Beta_t --> Epoch
`````
---

## Design for Scale 

The architecture of this repository is intentionally decoupled to facilitate rapid scaling across heterogeneous compute environments. By isolating the Clifford-encoded geometry from the SDE transition logic, the system allows for independent optimization of the data representation and the generative physics.

- **Compute Portability**: Because the engine is built on zero-dependency C++17 with raw buffer management, the transition kernels in Diffusion_model.cpp are immediately compatible with SIMD vectorization (AVX-512) and custom CUDA/Triton kernels.

- **Architectural Flexibility**: The use of functional hooks in main.cpp for epsilon-prediction means the current CNN-based predictor can be swapped for a Diffusion Transformer (DiT) or a State-Space Model (SSM) backend without altering the underlying Gaussian Diffusion SDEs.

- **Memory Efficiency**: The hybrid FP16/Huffman serialization pipeline is designed for high-bandwidth memory (HBM) constraints, ensuring that the latent field remains compact during large-scale distributed training.


The analytic gradient derivations in NormalDist.cpp provide a blueprint for moving beyond Gaussian noise into more complex Non-Euclidean Diffusion or Flow-Matching paradigms, ensuring the project remains at the frontier of generative research.

---

## Technical Specifications & Hardware Alignment

The engine is engineered for high-performance execution on ARM-based Unified Memory Architectures, specifically optimized for the Apple M4 Silicon ecosystem. By leveraging the M4's high-bandwidth HBM and advanced branch prediction, the pipeline achieves ultra-low latency inference for latent reconstruction.

| Category | Specification | Implementation Detail | M4 Silicon Advantage |
| :--- | :--- | :--- | :--- |
| **Numeric Precision** | `FP16` / `FP32` Hybrid | Manual IEEE 754 conversion logic | Native half-precision via NEON/AMX |
| **Latent Geometry** | Clifford Multivector | 8D (Compressed) / 64D (Full) | Optimized L1/L2 cache locality |
| **SDE Physics** | Discrete Markov Chain | $T = 1000$ timesteps | High IPC for serial denoising |
| **Probabilistic Kernel** | Analytic Gaussian PDF | Derived $\frac{\partial \mathcal{L}}{\partial \mu}$ and $\frac{\partial \mathcal{L}}{\partial \sigma}$ | Predictable branching for deep pipelines |
| **Optimization** | Adaptive Momentum | Custom C++ Adam Optimizer | Zero-copy Unified Memory access |
| **Serialization** | Huffman + BPE | Frequency-based bit-packing | Reduced I/O overhead on HBM |

### Hardware-Specific Optimization Note:

While the current implementation uses standard C++ loops, the memory layout is SIMD-ready. On M4 hardware, the clamp_vector and compute_mean_variance functions are prime candidates for Auto-Vectorization, allowing the processor to handle 128-bit or 256-bit wide chunks of the latent field in a single cycle.

## Usage

Build and run the project using CMake. Supply a text dataset file and specify an output file for compressed Huffman encoded embeddings. Training runs for a configurable number of epochs with beta scheduling dynamically adjusting noise levels.

```bash
chmod +x build_and_run.sh
./build_and_run.sh input_data.txt output_compressed.huff
```

⸻
Catherine Earl
MIT License 2026

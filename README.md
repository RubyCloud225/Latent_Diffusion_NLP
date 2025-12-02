# Latent Diffusion NLP

A high-performance latent diffusion pipeline for natural language processing tasks, combining tokenization, embedding, compression, and advanced diffusion-based generative modeling. This project demonstrates a full end-to-end approach from raw text data through learned latent representations to probabilistic diffusion training and compression.

---

## Overview

This project implements a modern latent diffusion framework designed to:

- Tokenize raw textual data with a **Byte-Pair Encoding (BPE)** tokenizer.
- Embed tokens into dense vector representations.
- Compress embeddings using **Clifford Algebra-based compression** followed by **Huffman coding** for efficient storage.
- Train a **Gaussian diffusion model** to learn noise-aware latent representations with adaptive noise scheduling.
- Use a **BetaSchedule** to dynamically control noise parameters across training epochs, balancing reconstruction and noise regularization.
- Model noise predictions through a neural network embedded in the diffusion framework.
- Leverage probabilistic modeling via the **normal distribution** for likelihood-based training and gradient estimation.

---

## Architecture

### 1. Tokenization & Embedding

The input raw text is first tokenized using a **simple BPE tokenizer**, merging common character pairs to create a compact vocabulary.  
Each token is embedded into a 64-dimensional vector space through a **learned embedding layer** that hashes tokens to dense vectors.

### 2. Clifford Compression & Huffman Coding

Token embeddings are compressed into multivectors via a novel **Clifford algebra compression** technique, retaining geometric properties in a compact form.  
The compressed data is serialized and further compressed using a **Huffman encoding** scheme to reduce storage size while maintaining lossless fidelity.

### 3. Gaussian Diffusion Model

A **Gaussian Diffusion** process is applied in latent space, gradually adding Gaussian noise to latent embeddings over a sequence of timesteps.

\[
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

The reverse denoising process learns to predict the noise \(\epsilon\) added at each step, enabling reconstruction of clean latent embeddings from noisy inputs.

### 4. Beta Schedule

A **BetaSchedule** dynamically updates the noise variance \(\beta_t\) during training epochs using loss statistics:

\[
\beta_{epoch} = \beta_0 \left(1 - \frac{epoch}{total\_epochs}\right) + \frac{epoch}{total\_epochs}
\]

This annealing balances noise injection and reconstruction fidelity.

### 5. Noise Prediction Network

A neural network predicts the noise component \(\epsilon\) given a noisy latent vector and timestep. The network is trained to minimize the negative log likelihood under a normal distribution:

\[
\mathcal{L} = -\log p_\theta(y \mid \mu, \sigma) = -\log \mathcal{N}(y; \mu, \sigma^2)
\]

where \(\mu\) and \(\sigma\) are derived from the model’s predictions and diffusion schedule.

### 6. Training Loop

Training iterates over epochs, applying the forward diffusion and noise prediction. Losses computed via the **NormalDist** module are used to update the BetaSchedule. Model parameters are optimized with an **Adam optimizer**.

---

## Key Mathematical Components

### Normal Distribution Log Probability

\[
\log p(y \mid \mu, \sigma) = -\frac{1}{2} \left( \frac{y - \mu}{\sigma} \right)^2 - \log(\sigma) - \frac{1}{2} \log(2\pi)
\]

### Gradient Computation for Training

Gradients with respect to \(y, \mu, \sigma\) guide model parameter updates through backpropagation.

### Forward Diffusion Step

\[
x_t = x_{t-1} + \mathcal{N}\left(0, \beta_t I\right)
\]

### Reverse Diffusion Mean Estimate

\[
\mu_{t-1} = \frac{x_t - \beta_t \epsilon_\theta(x_t, t)}{\sqrt{1 - \beta_t}}
\]

---

## Architectural Diagrams

### Overall Pipeline

```mermaid
graph TD
    A[Raw Text Dataset]
    B[BPE Tokenizer]
    C[Embedding Layer]
    D[Clifford Compression]
    E[Huffman Coding]
    F[Compressed Dataset Storage]
    G[Gaussian Diffusion Training]
    H[Noise Prediction Neural Network]
    I[Beta Schedule]
    J[Model Parameters & Adam Optimizer]
    K[Trained Diffusion Model]

    A --> B --> C --> D --> E --> F
    F --> G
    G --> H
    H --> J
    J --> G
    G --> I
    I --> G
    G --> K

graph LR
    X_{t-1}[x_{t-1}]
    Beta_t[Noise Variance \u03B2_t]
    Noise[\u03B5 \sim \mathcal{N}(0,I)]
    X_t[x_t]

    X_{t-1} --> |Add noise| X_t
    Beta_t --> X_t
    Noise --> X_t
    X_t --> |Predict \u03B5_\u03B8(x_t,t)| PredictedEps
    PredictedEps --> |Reverse mean \u03BC_{t-1}| X_{t-1}'

graph LR
    Epoch[Epoch]
    BetaStart[\u03B2_0]
    BetaEnd[1.0]
    Beta_t[\u03B2_{epoch}]
    Losses[NLL, Entropy Loss]

    Epoch --> Beta_t
    BetaStart --> Beta_t
    BetaEnd --> Beta_t
    Losses --> Beta_t
    Beta_t --> Epoch

## Usage

Build and run the project using CMake. Supply a text dataset file and specify an output file for compressed Huffman encoded embeddings. Training runs for a configurable number of epochs with beta scheduling dynamically adjusting noise levels.

⸻


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

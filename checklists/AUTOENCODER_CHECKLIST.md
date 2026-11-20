# Image Autoencoder Checklist

**Track:** AE — Image Autoencoder (Roadmap Milestones 2 and 4)  
**Status:** Not started — no items are signed off.  
**Scope:** Train a deterministic, testable C++17 CPU reference encoder/decoder
for 32 × 32 RGB images, producing spatial image latents that can later be used
by the epsilon-prediction network and sampler.

Related work: [P0 production core](P0_CHECKLIST.md),
[NN and epsilon predictor](NN_CHECKLIST.md), [roadmap](../ROADMAP.md).

## Architecture and boundaries

Initial target:

```text
Image Tensor [N, 3, 32, 32]  (normalised to [-1, 1])
          |
       Encoder
          v
Latent Tensor [N, C, 8, 8]   (C is configurable, not yet fixed)
          |
       Decoder
          v
Reconstruction [N, 3, 32, 32]
```

The encoder and decoder share the production Tensor shape conventions and
neural-layer/optimiser interfaces. The autoencoder is **not** the epsilon
predictor: its reconstruction loss must not be confused with diffusion noise
prediction loss. The latent shape and scaling must be fixed and documented
before training the final spatial denoiser.

## AE.1 — Image input/output and data contracts (Milestone 2)

- [ ] Define channel-first [C,H,W] and [N,C,H,W] image Tensor contracts.
- [ ] Set initial image target to 32 × 32 RGB; document channel ordering.
- [ ] Implement a minimal image loader for a documented file format.
- [ ] Implement a minimal image writer for reconstruction output.
- [ ] Convert pixels reproducibly to floating point and normalise to [-1,1].
- [ ] Implement correct inverse denormalisation and pixel clipping.
- [ ] Validate dimensions, channel count, non-finite values and file errors.
- [ ] Implement fixed-size validation or explicitly documented resize policy.
- [ ] Add known-answer pixel-conversion and round-trip tests.
- [ ] Separate dataset/outputs from source-controlled files.

**Exit criterion:** A small image can be loaded, normalised, denormalised and
written with documented numerical and pixel-error tolerances.

## AE.2 — Architecture definition

- [ ] Define `ImageEncoder` and `ImageDecoder` public APIs.
- [ ] Choose and document initial latent channel count `C`.
- [ ] Define intermediate convolution/downsample and upsample shapes.
- [ ] Define the latent output policy (deterministic AE by default; VAE only
      after a separately reviewed design).
- [ ] Decide whether encoder latents require fixed scaling for downstream
      diffusion; document and test the convention.
- [ ] Specify trainable parameters, initialisation and gradient ownership.
- [ ] Add Doxygen annotations for shapes, precision, ownership and exceptions.

**Exit criterion:** The architecture and numerical contracts are documented
before implementing or training the model.

## AE.3 — Encoder forward and backward passes

- [ ] Implement image-to-latent forward pass for [N,3,32,32].
- [ ] Reuse validated NN.1 layers and parameter/gradient interfaces.
- [ ] Implement encoder backward propagation.
- [ ] Validate output shape [N,C,8,8] for batch sizes including 1.
- [ ] Test deterministic weights, inputs and outputs.
- [ ] Reject mismatched shapes and non-finite inputs.
- [ ] Add representative finite-difference gradient checks.

**Exit criterion:** The encoder produces expected latent shapes and gradients
with numerical tests passing.

## AE.4 — Decoder forward and backward passes

- [ ] Implement latent-to-image forward pass.
- [ ] Reuse validated upsampling/transpose-convolution components.
- [ ] Define output activation and [-1,1] numerical convention.
- [ ] Implement decoder backward propagation.
- [ ] Validate exact reconstructed output shape [N,3,32,32].
- [ ] Test deterministic forward output and invalid latent shapes.
- [ ] Add numerical gradient checks for decoder parameters.

**Exit criterion:** Decoder returns valid images with correct gradients,
shape and finite values.

## AE.5 — Reconstruction objective and training

- [ ] Implement documented reconstruction loss (begin with mean squared error).
- [ ] Validate loss against an independent small known-answer fixture.
- [ ] Propagate gradients through decoder and encoder.
- [ ] Integrate the validated NN.1 optimiser, zero-grad and gradient clipping.
- [ ] Implement a deterministic, seeded CPU mini-batch training loop.
- [ ] Establish a train/validation split with no example leakage.
- [ ] Log train and validation reconstruction loss per epoch.
- [ ] Test a tiny overfitting case to confirm the network can learn.
- [ ] Verify improvement over an untrained baseline.
- [ ] Guard against non-finite weights, gradients and losses.

**Exit criterion:** A small training run reproducibly reduces reconstruction
loss; independent validation metrics are recorded.

## AE.6 — Checkpoints and reproducibility

- [ ] Save encoder, decoder and optimiser parameters with model metadata.
- [ ] Record architecture, latent shape, scaling, seed and training step.
- [ ] Implement checkpoint load and schema/version validation.
- [ ] Support deterministic resume, including RNG and optimiser state.
- [ ] Reject corrupt/incompatible checkpoints with clear errors.
- [ ] Test checkpoint round-trip and resumed-training equivalence.

**Exit criterion:** Training can stop, reload and resume without unexplained
parameter, loss or numerical-state differences.

## AE.7 — Reconstruction CLI and evaluation (Milestone 4)

- [ ] Create a `latent_reconstruct` application/CLI.
- [ ] Accept input image, checkpoint and output path.
- [ ] Produce and save reconstructions without invoking training code.
- [ ] Save before/after samples for train and held-out images.
- [ ] Measure validation MSE/MAE and pixel-level error.
- [ ] Visually inspect held-out reconstructions for recognisable structure.
- [ ] Verify repeatability from the same checkpoint and input.
- [ ] Add error-path tests for missing files and bad checkpoint versions.

**Exit criterion:** Unseen 32 × 32 images can be reconstructed into
recognisable outputs with measured, repeatable validation results.

## AE.8 — Handoff to latent diffusion

- [ ] Freeze/document latent tensor shape, channel ordering and scaling.
- [ ] Verify deterministic `encode(image)` and `decode(latent)` interfaces.
- [ ] Confirm denoiser input/output shapes exactly match latent dimensions.
- [ ] Define whether autoencoder weights stay frozen during denoiser training.
- [ ] Test encoded latents through P0 forward diffusion for shape/finiteness.
- [ ] Add a decode-from-latent integration test for the eventual generator.
- [ ] Record CPU tests, Release validation and documentation generation
      separately; do not infer Metal parity.

**Exit criterion:** A documented and tested autoencoder checkpoint and latent
contract are ready for NN.2 epsilon-predictor training and NN.3 image generation.

## Dependencies and sign-off rules

- NN.1 trainable operations and gradients are prerequisites to meaningful
  AE training; image I/O and architecture design can proceed independently.
- AE.1–AE.7 can be developed in parallel with NN.2 on synthetic latent data,
  but **full image generation requires AE.8, trained epsilon prediction, and
  the completed reverse sampler**.
- Each section remains unchecked until the code is reviewed and test results
  are recorded. Include Doxygen for every public interface.
- Preserve the experimental Clifford path as an independent later comparison;
  do not make it a prerequisite for the standard autoencoder baseline.

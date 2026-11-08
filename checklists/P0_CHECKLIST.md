# P0 — Production Core Checklist

## P0.1 — Repository Cleanup

- [x] Remove tracked build artefacts
- [x] Confirm `build/` is ignored
- [x] Confirm legacy code remains isolated
- [x] Confirm experimental Clifford code remains isolated
- [x] Confirm production source layout is clean

## P0.2 — Build System

- [x] Create `latent_core` library target
- [x] Add core source files to `latent_core`
- [x] Add diffusion source files to `latent_core`
- [x] Add public include paths
- [x] Keep experimental target separate
- [x] Keep Metal target separate
- [x] Add compiler warnings
- [x] Validate Debug build
- [x] Validate Release build

## P0.3 — Test Harness

- [ ] Add production-core test targets
- [ ] Add core test directory
- [ ] Add diffusion test directory
- [ ] Add CTest labels
- [ ] Add `p0` label
- [ ] Add `core` label
- [ ] Add `diffusion` label
- [ ] Keep experimental tests separately labelled
- [ ] Keep Metal tests separately labelled
- [ ] Allow P0 tests to run independently

## P0.4 — Tensor

- [ ] Implement Tensor constructor
- [ ] Implement contiguous storage
- [ ] Implement shape storage
- [ ] Implement rank
- [ ] Implement total size calculation
- [ ] Implement mutable indexing
- [ ] Implement const indexing
- [ ] Implement bounds validation
- [ ] Implement raw data access
- [ ] Implement fill operation
- [ ] Implement shape comparison
- [ ] Implement finite-value validation
- [ ] Reject invalid dimensions
- [ ] Add Tensor unit tests

## P0.5 — Random Generator

- [ ] Implement seeded `RandomGenerator`
- [ ] Implement deterministic uniform sampling
- [ ] Implement deterministic Gaussian sampling
- [ ] Implement Gaussian Tensor generation
- [ ] Validate identical seeds reproduce identical sequences
- [ ] Validate different seeds produce different sequences
- [ ] Validate generated values are finite
- [ ] Add RNG unit tests

## P0.6 — Noise Schedule

- [ ] Implement linear schedule configuration
- [ ] Validate timestep count
- [ ] Validate beta range
- [ ] Validate beta start/end ordering
- [ ] Precompute beta values
- [ ] Precompute alpha values
- [ ] Precompute cumulative alpha-bar values
- [ ] Precompute `sqrt(alpha_bar)`
- [ ] Precompute `sqrt(1 - alpha_bar)`
- [ ] Add timestep bounds validation
- [ ] Add finite-value validation
- [ ] Add NoiseSchedule unit tests

## P0.7 — Forward Diffusion

- [ ] Implement `q_sample`
- [ ] Accept explicit input Tensor
- [ ] Accept explicit timestep
- [ ] Accept explicit noise Tensor
- [ ] Validate input/noise shapes
- [ ] Implement cumulative-alpha forward equation
- [ ] Preserve input Tensor
- [ ] Preserve supplied noise Tensor
- [ ] Add known-answer test
- [ ] Add zero-noise test
- [ ] Add zero-input test
- [ ] Add timestep boundary tests
- [ ] Add deterministic repeat test
- [ ] Add NaN/Inf validation tests

## P0.8 — Reverse Diffusion Mathematics

- [ ] Implement posterior variance coefficients
- [ ] Implement reverse-process coefficients
- [ ] Implement reverse mean calculation
- [ ] Accept externally supplied predicted epsilon
- [ ] Validate predicted-epsilon shape
- [ ] Handle final reverse timestep correctly
- [ ] Add known-answer reverse test
- [ ] Add deterministic reverse-mean test
- [ ] Add finite-value tests
- [ ] Add posterior variance tests

## P0.9 — Sampler

- [ ] Implement one reverse sampling step
- [ ] Integrate `RandomGenerator`
- [ ] Accept predicted epsilon explicitly
- [ ] Add stochastic noise only when required
- [ ] Remove placeholder random epsilon behaviour
- [ ] Preserve Tensor shape
- [ ] Validate terminal timestep behaviour
- [ ] Validate deterministic seeded sampling
- [ ] Validate different seeds alter stochastic steps
- [ ] Add sampler unit tests

## P0.10 — Configuration

- [ ] Populate `configs/tiny_image.toml`
- [ ] Define image width
- [ ] Define image height
- [ ] Define image channels
- [ ] Define latent dimensions
- [ ] Define diffusion timesteps
- [ ] Define diffusion schedule
- [ ] Define beta start
- [ ] Define beta end
- [ ] Define deterministic seed

## P0.11 — Build and Test Scripts

- [ ] Implement `scripts/build.sh`
- [ ] Configure project from script
- [ ] Build project from script
- [ ] Enable tests from script
- [ ] Update `scripts/test.sh`
- [ ] Add P0-only test execution
- [ ] Enable `--output-on-failure`
- [ ] Ensure script exits on build failure
- [ ] Ensure script exits on test failure

## P0.12 — Regression Validation

- [ ] Run complete P0 test suite
- [ ] Re-run FP16 tests
- [ ] Re-run Clifford projection tests
- [ ] Re-run Clifford gradient tests
- [ ] Re-run Clifford Adam training test
- [ ] Re-run Metal projection tests where supported
- [ ] Confirm no existing validated behaviour regressed

## P0.13 — Documentation

- [ ] Update `STATUS.md`
- [ ] Update completed P0 items in `ROADMAP.md`
- [ ] Add P0 changes to `CHANGELOG.md`
- [ ] Record P0 test results
- [ ] Record CPU validation result
- [ ] Record Metal validation result where applicable

## P0.14 — Clean Clone Validation

- [ ] Clone repository into a clean directory
- [ ] Configure without local state
- [ ] Build successfully
- [ ] Run P0 tests successfully
- [ ] Run regression tests successfully
- [ ] Confirm no generated build files are tracked
- [ ] Confirm no hidden local dependencies are required
- [ ] Confirm repository remains clean after build
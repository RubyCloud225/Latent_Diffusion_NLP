# Production API Documentation

The project uses [Doxygen](https://www.doxygen.nl/) to generate browsable
API documentation from C++ public headers. Doxygen is **optional** and is not
required for normal CPU builds, tests, or Metal builds.

## Generate documentation (macOS)

Install Doxygen if needed:

```bash
brew install doxygen
```

From the repository root:

```bash
bash scripts/run_documentation.sh
bash scripts/run_documentation.sh --open
```

The script uses a dedicated `build-docs/` CMake directory so your
ordinary `build/` and `scripts/test.sh` configuration remain unchanged.
The HTML output is at `build-docs/docs/html/index.html`.

For advanced/manual use:

```bash
cmake -S . -B build-docs -DBUILD_DOCS=ON -DBUILD_TESTING=OFF -DBUILD_METAL_TESTS=OFF
cmake --build build-docs --target docs
```

Use `DOCS_BUILD_DIR` to choose a different documentation build directory.

## Scope

The documentation target reads `latent/core/` and `latent/diffusion/`.
These public headers document the supported Tensor, RandomGenerator,
NoiseSchedule and GaussianDiffusion interfaces. The experimental Clifford
and legacy NLP implementations are outside this production API scope.

When extending an API, add Doxygen `@brief`, `@param`, `@return`,
`@throws` (where applicable), and numerical/ownership constraints.
The generated HTML is build output and must not be committed.

## Verification

The target is available only when `BUILD_DOCS=ON` and Doxygen is installed.
A successful CTest run does **not** verify documentation generation; run the
`docs` build separately.

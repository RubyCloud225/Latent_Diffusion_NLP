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
cmake -S . -B build -DBUILD_DOCS=ON -DBUILD_METAL_TESTS=OFF
cmake --build build --target docs
open build/docs/html/index.html
```

For users of an existing build directory, rerunning CMake with
`-DBUILD_DOCS=OFF` disables the documentation target.

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

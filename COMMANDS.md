# Commands

Common commands for the C++17 production latent-diffusion project. Run from
the repository root. Invoke scripts with `bash`; executable permissions are
not required.

## Requirements

CMake 3.16+, a C++17 compiler, and optionally Doxygen (for HTML documentation)
or Apple's Metal Toolchain (for Metal tests).

## Get the latest source

```bash
git status --short
git pull
```

## Build

```bash
bash scripts/build.sh
BUILD_TYPE=Release BUILD_DIR="$PWD/build-release" bash scripts/build.sh
```

The default is Debug in `build/`; optional Metal targets are off by default.
Use separate directories for Debug, Release and documentation builds.

## Test

```bash
bash scripts/test.sh --p0
bash scripts/test.sh --all
bash scripts/test.sh --core
bash scripts/test.sh --diffusion
bash scripts/test.sh --experimental
bash scripts/test.sh --nn
```

For optional Metal validation (requires an installed Apple Metal Toolchain):

```bash
bash scripts/test.sh --metal
```

To run a single registered CTest test or inspect failures:

```bash
ctest --test-dir build -R reverse_diffusion --output-on-failure
ctest --test-dir build -L p0 --output-on-failure
ctest --test-dir build --output-on-failure
```

## Doxygen API documentation

Install Doxygen on macOS if needed:

```bash
brew install doxygen
```

The documentation script configures `build-docs/` independently from the
normal `build/` directory.

```bash
bash scripts/run_documentation.sh
bash scripts/run_documentation.sh --open
bash scripts/run_documentation.sh --help
```

Optional custom output directory:

```bash
DOCS_BUILD_DIR="$PWD/custom-docs-build" bash scripts/run_documentation.sh
```

HTML index: `build-docs/docs/html/index.html`.
See [docs/DOXYGEN.md](docs/DOXYGEN.md) for details.

## Repository layout

| Path | Purpose |
| --- | --- |
| `latent/core/` | Public Tensor and random-generator APIs |
| `latent/diffusion/` | Public diffusion and schedule APIs |
| `src/core/`, `src/diffusion/` | Production implementation |
| `tests/core/`, `tests/diffusion/` | Numerical unit tests |
| `tests/experimental/` | Isolated experimental tests |
| `checklists/P0_CHECKLIST.md` | Milestone acceptance criteria |

## Contributions and sign-off

Consult [CONTRIBUTION.md](CONTRIBUTION.md) before submitting changes.
Record verified test results in `CHANGELOG.md` and the appropriate checklist.
CPU CTest results do not establish Metal, Release, clean-clone or Doxygen
validation. Do not commit generated build or HTML documentation artifacts.

```bash
git diff
git status --short
git add <specific-files>
git commit -m "Describe the validated change"
git push
```

#!/usr/bin/env bash
# Generate API documentation without modifying the normal test build.
# Usage: bash scripts/run_documentation.sh [--open|--help]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_BUILD_DIR="${DOCS_BUILD_DIR:-${ROOT_DIR}/build-docs}"
mode="${1:-}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: bash scripts/run_documentation.sh [--open|--help]" >&2
  exit 2
fi
case "${mode}" in
  ""|--open) ;;
  --help|-h)
    cat <<'USAGE'
Usage: bash scripts/run_documentation.sh [--open|--help]
Generate production C++ Doxygen HTML into build-docs/ (not build/).
  --open   Open the HTML index in the system browser.
  --help   Print this help.
Set DOCS_BUILD_DIR to override the documentation build location.
USAGE
    exit 0
    ;;
  *) echo "Unknown option: ${mode}" >&2; exit 2 ;;
esac

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake is required." >&2
  exit 1
fi
if ! command -v doxygen >/dev/null 2>&1; then
  echo "Doxygen is required. On macOS: brew install doxygen" >&2
  exit 1
fi

echo "==> Configuring Doxygen in ${DOCS_BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${DOCS_BUILD_DIR}" \
  -DBUILD_DOCS=ON -DBUILD_TESTING=OFF -DBUILD_METAL_TESTS=OFF
echo "==> Generating documentation"
cmake --build "${DOCS_BUILD_DIR}" --target docs

INDEX="${DOCS_BUILD_DIR}/docs/html/index.html"
if [[ ! -f "${INDEX}" ]]; then
  echo "Documentation index missing: ${INDEX}" >&2
  exit 1
fi
echo "==> Generated: ${INDEX}"

if [[ "${mode}" == "--open" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${INDEX}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${INDEX}"
  else
    echo "No browser opener found; open ${INDEX} manually." >&2
    exit 1
  fi
fi

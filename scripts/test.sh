#!/usr/bin/env bash
# Build, then run all CTest tests or one labelled group.
# Usage: ./scripts/test.sh [--all|--p0|--core|--diffusion|--experimental|--metal]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
mode="${1:---all}"

case "${mode}" in
    --all) label="" ;;
    --p0) label="p0" ;;
    --core) label="core" ;;
    --diffusion) label="diffusion" ;;
    --experimental) label="experimental" ;;
    --metal) label="metal" ;;
    --nn) label="nn" ;;
    *) echo "Usage: $0 [--all|--p0|--core|--diffusion|--experimental|--metal|--nn]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Only one test selection option is supported" >&2
    exit 2
fi

if [[ "${mode}" == "--metal" ]]; then
    export BUILD_METAL_TESTS=ON
fi
bash "${ROOT_DIR}/scripts/build.sh"
echo "==> Running ${mode} tests"
ctest_args=(--test-dir "${BUILD_DIR}" --output-on-failure --no-tests=error)
if [[ -n "${label}" ]]; then
    ctest_args+=(-L "${label}")
fi
ctest "${ctest_args[@]}"
echo "==> Tests passed"

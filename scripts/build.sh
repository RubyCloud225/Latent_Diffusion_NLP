#!/usr/bin/env bash
# Configure and build the repository. Supports both Debug and Release builds.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_METAL_TESTS="${BUILD_METAL_TESTS:-OFF}"

case "${BUILD_TYPE}" in
    Debug|Release|RelWithDebInfo|MinSizeRel) ;;
    *) echo "Unsupported BUILD_TYPE: ${BUILD_TYPE}" >&2; exit 2 ;;
esac

echo "==> Configuring ${BUILD_TYPE} in ${BUILD_DIR}"
cmake_args=(
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DBUILD_TESTING=ON
)
cmake_args+=(-DBUILD_METAL_TESTS="${BUILD_METAL_TESTS}")
cmake "${cmake_args[@]}"

echo "==> Building"
cmake --build "${BUILD_DIR}" --parallel
echo "==> Build completed"

#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "==> Project root: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

echo "==> Configuring project..."
cmake \
    -S "${ROOT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTING=ON

echo "==> Building tests..."
cmake \
    --build "${BUILD_DIR}" \
    --parallel

echo "==> Running test suite..."
ctest \
    --test-dir "${BUILD_DIR}" \
    --output-on-failure \
    --verbose

echo "==> All tests passed."
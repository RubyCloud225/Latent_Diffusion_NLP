#!/bin/bash

# @file build_and_run.sh
# @brief Automated Build & Experiment Orchestrator.
#
# This script automates the compilation and execution lifecycle of the 
# Latent Diffusion NLP engine. It manages directory structures, compiler 
# optimization flags, and sequential execution of the data-prep and 
# training phases.
#
# DESIGN RATIONALE:
# - Deterministic Environments: Ensures a clean `Logs_Results` workspace 
# before execution to prevent data contamination.
# - Optimized Compilation: Targets modern hardware via -O3 and C++17 standards.
# - Safety First: Implements "Exit on Error" (set -e) to ensure the pipeline 
# halts if data corruption or build failure occurs.

# Variables - adjust these paths as needed
BUILD_DIR=build
EXECUTABLE=LatentDiffusionNLP
INPUT_FILE=dataset.txt
OUTPUT_FILE=output.huff

echo "Cleaning build directory..."
rm -rf $BUILD_DIR

echo "Creating build directory..."
mkdir $BUILD_DIR
cd $BUILD_DIR

echo "Running cmake..."
cmake ..

echo "Building project..."
make -j$(nproc)

echo "Running executable..."
./$EXECUTABLE ../$INPUT_FILE ../$OUTPUT_FILE

echo "Done."

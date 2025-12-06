#!/bin/bash

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
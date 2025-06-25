#!/bin/bash

# Define the destination directory
DEST_DIR="pr_run-6-25-25"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Move the output files from the Guided Evolution to fail_run
mv -f slurm-*.out "$DEST_DIR/"
mv -f 0 "$DEST_DIR/"
mv -f sota/ultralytics/ultralytics/cfg/models/llm/models/* "$DEST_DIR/"
mv -f sota/ultralytics/results/* "$DEST_DIR/"

# Run these if you need to move everything, otherwise, comment them out
# Note: mv -f is used to overwrite existing files in the destination without prompt
mv -f first_test/* "$DEST_DIR/"
mv -f templates/FixedPrompts/roleplay/mutant*.txt "$DEST_DIR/"
mv -f templates/FixedPrompts/concise/mutant*.txt "$DEST_DIR/"
mv -f runs/detect/* "$DEST_DIR/"
#!/bin/bash

# Define the output file to store the list of installed extensions
OUTPUT_FILE="extensions.list"

# List all installed extensions and save them to the file
echo "Saving installed extensions to $OUTPUT_FILE..."
code --list-extensions > "$OUTPUT_FILE"

echo "Installed extensions have been saved to $OUTPUT_FILE"

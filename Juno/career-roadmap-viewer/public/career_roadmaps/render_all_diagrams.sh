#!/bin/bash

# Script to render all Mermaid diagrams to PNG and SVG formats
# Uses mermaid-cli (mmdc) which should be installed via npm

# Set script to exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define output formats
FORMATS=("png" "svg")

# Check if mmdc is installed
if ! command -v mmdc &> /dev/null; then
    echo "Error: mermaid-cli (mmdc) is not installed or not in PATH."
    echo "Please install it using: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# Display version information
echo "Using mermaid-cli version: $(mmdc --version 2>&1 | head -n 1)"
echo "Rendering diagrams..."
echo ""

# Find all career paths automatically
CAREER_PATHS=()
for dir in "$SCRIPT_DIR"/*/; do
    # Get the directory name without the trailing slash
    dir_name=$(basename "${dir%/}")

    # Skip non-career path directories
    if [[ "$dir_name" == "rendered_diagrams" || "$dir_name" == "node_modules" ]]; then
        continue
    fi

    # Check if this directory has a diagrams folder with a career_path.mmd file
    if [[ -f "$dir/diagrams/career_path.mmd" ]]; then
        CAREER_PATHS+=("$dir_name")
    fi
done

# Process each career path
for path in "${CAREER_PATHS[@]}"; do
    echo "Processing $path career path..."

    # Input file path
    input_file="$SCRIPT_DIR/$path/diagrams/career_path.mmd"

    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file $input_file does not exist. Skipping."
        continue
    fi

    # Generate diagrams in different formats
    for format in "${FORMATS[@]}"; do
        # Output directly to the diagrams folder
        output_file="$SCRIPT_DIR/$path/diagrams/career_path.$format"
        echo "  Generating $format diagram..."

        # Add specific options based on format
        if [ "$format" == "png" ]; then
            mmdc -i "$input_file" -o "$output_file" -b white -w 4096 -H 4096 -t forest
        else
            mmdc -i "$input_file" -o "$output_file" -b white -t forest
        fi
    done

    echo "  Done with $path."
    echo ""
done

echo "All diagrams have been rendered successfully!"
echo "Diagrams are saved in each career path's diagrams folder."
echo ""
echo "You can view these diagrams in the diagram_viewer.html file."

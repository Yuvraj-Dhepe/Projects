#!/bin/bash

# Define the career paths
CAREER_PATHS=(
    "engineering"
    "food_inspector"
    "armed_forces"
    "bsc_streams"
    "isro"
    "teaching"
    "police"
    "air_hostess"
)

# Process each career path
for path in "${CAREER_PATHS[@]}"; do
    echo "Updating README for $path career path..."
    
    # README file path
    readme_file="/home/yuvidh/work/Projects/Juno/career_roadmaps/$path/README.md"
    
    # Check if README file exists
    if [ ! -f "$readme_file" ]; then
        echo "Warning: README file $readme_file does not exist. Skipping."
        continue
    fi
    
    # Add links to rendered diagrams
    sed -i "/View the full diagram here/a\\
\\
**Rendered Diagrams:**\\
- [PNG Version](../rendered_diagrams/${path}_career_path.png)\\
- [SVG Version](../rendered_diagrams/${path}_career_path.svg)\\
" "$readme_file"
    
    echo "  Done updating $path README."
    echo ""
done

echo "All README files have been updated successfully!"

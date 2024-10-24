#!/bin/bash

# File containing the list of extensions
EXTENSIONS_FILE="extensions.list"

# Check if the file exists
if [ ! -f "$EXTENSIONS_FILE" ]; then
    echo "Extensions file $EXTENSIONS_FILE not found!"
    exit 1
fi

# Read the extensions list and install each extension
while IFS= read -r extension; do
    if [[ -n "$extension" ]]; then  # Make sure it's not an empty line
        echo "Installing extension: $extension"
        code --install-extension "$extension"
    fi
done < "$EXTENSIONS_FILE"

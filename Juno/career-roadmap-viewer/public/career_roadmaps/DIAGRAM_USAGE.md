# Career Roadmap Diagrams

This directory contains Mermaid diagrams for various career paths. The diagrams are stored as `.mmd` files in each career path's `diagrams` directory.

## Viewing Diagrams

You can view the diagrams using the `diagram_viewer.html` file. Simply open this file in your web browser to see all career path diagrams.

### Features

- **View PNG Diagrams**: Shows all career path diagrams in PNG format
- **View SVG Diagrams**: Shows all career path diagrams in SVG format
- **Render All Diagrams**: Provides instructions for rendering all diagrams using the command-line tool
- **Automatic Career Path Detection**: Automatically detects and displays all career paths with diagram files

## Rendering Diagrams

The diagrams can be rendered to PNG and SVG formats using the `render_all_diagrams.sh` script. This script uses [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) to render the diagrams.

### Prerequisites

- [Node.js](https://nodejs.org/) (required for mermaid-cli)
- mermaid-cli: Install using `npm install -g @mermaid-js/mermaid-cli`

### Usage

1. Make sure you have mermaid-cli installed
2. Run the script: `./render_all_diagrams.sh`
3. The rendered diagrams will be saved in each career path's `diagrams` directory as:
   - `career_path.png`
   - `career_path.svg`

## Creating New Diagrams

To create a new diagram for a career path:

1. Create a new directory for the career path (if it doesn't exist)
2. Create a `diagrams` subdirectory
3. Create a `career_path.mmd` file in the diagrams directory
4. Write your Mermaid diagram code in the file
5. Run the `render_all_diagrams.sh` script to render the diagram

The diagram viewer will automatically detect and display the new career path diagram.

## Mermaid Syntax

Mermaid diagrams use a simple syntax to define diagrams. Here's a basic example:

```
graph TD
    A[Start] --> B[Process]
    B --> C[End]
```

For more information on Mermaid syntax, visit the [Mermaid documentation](https://mermaid.js.org/intro/).

## Troubleshooting

If you encounter issues with rendering diagrams:

1. Check that mermaid-cli is installed correctly
2. Verify that your Mermaid syntax is valid
3. Make sure your career path directory structure follows the pattern:
   ```
   career_path_name/
     └── diagrams/
         ├── career_path.mmd
         ├── career_path.png (generated)
         └── career_path.svg (generated)
   ```
4. Check the browser console for any JavaScript errors when using the diagram viewer

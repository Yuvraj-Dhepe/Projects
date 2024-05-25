# Coding Conventions for a Computer Vision Project

## General Naming Conventions

- **Variables and Functions**: Use `snake_case`
- **Classes**: Use `PascalCase`
- **Constants**: Use `UPPERCASE_WITH_UNDERSCORES`

## File Naming Conventions

- Use `snake_case` for filenames
- Use descriptive names that convey the purpose

## Directory Structure

- **Data**: `data/raw/`, `data/processed/`, `data/annotations/`, `data/splits/`
- **Source Code**: `src/data/`, `src/models/`, `src/training/`, `src/inference/`, `src/utils/`
  - However this can change on the base of Project System Design
  - Always write a Basic System Design for every project
- **Notebooks**: Use a numbered prefix and descriptive name (e.g., `01_data_exploration.ipynb`)

## Documentation

- **Docstrings**: Use a consistent format, including summary, parameters, return values, and exceptions
  - Currently in use [custom_docstring](./custom_docstring.mustache)
- **Comments**: Write clear, concise comments explaining the purpose of complex or non-obvious code sections

## Functions and Methods

- Use descriptive names that clearly indicate the function's purpose
- Use type hints for parameters and return values

## Classes

- Use PascalCase for class names
- Organize related methods within the same class

## Constants

- Use uppercase letters with underscores for constants

## Imports

- Group imports in the following order: standard library, third-party libraries, local modules
- Use absolute imports when possible

## Indentation and Spacing

- Use 4 spaces per indentation level
- Use formatter & a .settings of vscode for formatting


# Installation Guide

## Prerequisites
- Python 3.8 or higher
- MATLAB (for legacy tools)

## Python Setup
1. Create a virtual environment (recommended):
   ```pwsh
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```pwsh
   pip install -r requirements.txt
   ```
   or use Poetry/UV with `pyproject.toml`.

## MATLAB Setup
- Open MATLAB and add relevant folders to your path.

## Testing
- Run Python tests with:
   ```pwsh
   pytest
   ```

## Troubleshooting
- Ensure all dependencies are installed.
- Check each module's README for specific setup notes.

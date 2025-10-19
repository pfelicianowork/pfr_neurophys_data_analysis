# PFR Neurophys Data Analysis

This repository contains tools and scripts for analyzing neurophysiology data, including spike analysis, position analysis, SWR detection, and Open Ephys data loading. It integrates Python and MATLAB modules for flexible workflows.

## Folder Overview
- **oe_old_analysis_tools/**: Legacy MATLAB and Python tools for Open Ephys data conversion and analysis.
- **oe_python_tools/**: Modern Python package for Open Ephys data analysis and streaming.
- **open_ephys_loader/**: Python utilities for loading and processing Open Ephys data.
- **position_analysis/**: Tools for position data analysis, interpolation, and linearization.
- **spike_analysis/**: Spike extraction, analysis, and documentation.
- **swr_detection/**: SWR event detection and analysis.

## Installation
1. Clone the repository.
2. Install Python 3.8+ and MATLAB (if using MATLAB scripts).
3. Install Python dependencies:
   ```pwsh
   pip install -r requirements.txt
   ```
   or use `pyproject.toml` with Poetry/UV.

## Usage
Refer to each module's README for specific instructions and examples.

## Documentation
- Each submodule contains its own `README.md` and guides.
- See `IMPLEMENTATION_SUMMARY.md` and other guides in relevant folders.

## Contributing
See `CONTRIBUTING.md` for guidelines.

## License
See `LICENSE` for terms.

## Contact
For questions, contact the repository maintainer.

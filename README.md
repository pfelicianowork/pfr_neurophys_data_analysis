# PFR Neurophys Data Analysis

This repository provides a comprehensive suite of tools and scripts for the analysis of neurophysiology data, primarily focusing on data acquired from Open Ephys systems. The project aims to facilitate research in neurophysiology by offering modules for spike analysis, animal position tracking, Sharp-Wave Ripple (SWR) detection, and efficient Open Ephys data loading. It leverages both Python and MATLAB to provide flexible and powerful data processing and visualization workflows.

## Folder Overview
- **oe_old_analysis_tools/**: Contains legacy MATLAB and Python scripts primarily used for Open Ephys data conversion and initial analysis. This section is maintained for compatibility with older workflows.
- **oe_python_tools/**: A modern Python package designed for advanced Open Ephys data analysis, control, and streaming functionalities. This is intended for current and future development.
- **open_ephys_loader/**: Dedicated Python utilities for robust loading, preprocessing, and handling of Open Ephys data, including lazy loading and LFP processing.
- **position_analysis/**: Tools specifically for the analysis of animal position data, including interpolation, linearization, and kinematic calculations, crucial for behavioral correlation with neural activity.
- **spike_analysis/**: Modules for the extraction, sorting, and detailed analysis of neuronal spike data, along with comprehensive documentation and guides.
- **swr_detection/**: Implements algorithms and pipelines for the detection and analysis of Sharp-Wave Ripples (SWRs), significant events in neural activity.
- **notebooks/**: Jupyter notebooks for interactive data exploration, analysis, visualization, and demonstration of various functionalities within the project.

## Installation
1. Clone the repository to your local machine.
2. Ensure you have Python 3.8+ installed. MATLAB is also required if you plan to utilize the legacy MATLAB scripts.
3. Install Python dependencies by running:
   ```pwsh
   pip install -r requirements.txt
   ```
   Alternatively, if you use Poetry or UV, you can install dependencies via `pyproject.toml`.

## Usage
Each module within this repository includes its own `README.md` file with specific instructions, examples, and usage guidelines. Jupyter notebooks in the `notebooks/` directory also provide practical demonstrations.

## Documentation
- Detailed `README.md` files and guides are available within each submodule.
- Refer to `IMPLEMENTATION_SUMMARY.md`, `MUA_EXTRACTION_GUIDE.md`, and `QUICK_REFERENCE.md` in the `spike_analysis/` directory for in-depth information on spike-related functionalities.

## Contributing
Please see `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License
This project is licensed under the terms specified in the `LICENSE` file.

## Contact
For any questions, issues, or further information, please contact the repository maintainer.

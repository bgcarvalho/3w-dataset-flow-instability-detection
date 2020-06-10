# 3W dataset - Flow instability detection

Machine learning applied to Flow Instability problem from the 3W dataset.

## Requirements

- Python3
- pip
- virtualenv
- 7zip

## Install

Clone the repository:

`git clone https://github.com/bgcarvalho/3w-dataset-flow-instability-detection.git`

Enter the directory:

`cd 3w-dataset-flow-instability-detection`

Then run install.sh:

`./install.sh`

This will clone 3W Dataset, extract CSV files and adjust relative path. It
creates a virtual environment and installs dependencies.

## Run

Set algorithm parameters (window size, step, etc) in the main function,
and then run the following command:

`./venv/bin/python3 experiment.py`

## License

MIT.

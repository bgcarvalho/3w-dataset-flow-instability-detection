#!/bin/bash

# download size: 816 MiB
git clone https://github.com/ricardovvargas/3w_dataset

# total disk size: 6 GB
7z x 3w_dataset/data/data.7z.001 -o3w_dataset

# python env
python3.8 -m virtualenv venv

# activate env
source ./venv/bin/activate

# install deps
python3 -m pip install -r requirements.txt

# compile Cython
cythonize -a -i utils/libutils.pyx

# keep env activated

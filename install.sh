#!/bin/bash

# download size: 816 MiB
git clone https://github.com/ricardovvargas/3w_dataset

# totapl disk size: 6 GB
7z x 3w_dataset/data/data.7z.001 -o3w_dataset

# python env
python3 -m virtualenv venv
./venv/bin/python3 -m pip install -r requirements.txt

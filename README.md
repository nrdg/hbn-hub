# hbn-hub

A place to put things we use to analyze data from [HBN](https://healthybrainnetwork.org/).

## Usage

Install the conda env that includes dependencies we use:

    conda create -n -f environment.yml
    conda activate hbn

You are now working in an environment with the dependencies installed!

Initialize the precommit hook that strips jupyter notebooks of their outputs:

    pre-commit install

# CUDA Project

## What is this?

The goal of this project is to compare matrix multiplication performance on varying matrix sizes on a CPU vs GPU

## Setup

Compile and run
```
nvcc -x cu main.cpp matmul.cpp matmul.cu -o matrix_multiply
./matrix_multiply
```

Visualize results
```
python -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn
python3 viz.py
```
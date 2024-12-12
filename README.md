# CUDA Project

## What is this?

The goal of this project is to compare matrix multiplication performance on varying matrix sizes on a CPU vs GPU

## Setup

Clone repository
```
git clone https://github.com/puravparab/CUDA_project.git
cd CUDA_project
```

Compile
```
nvcc -x cu main.cpp matmul.cpp matmul.cu -o matmul
```
Run executable
```
./matmul
```

## Visualize results

Create virtual environment for python
```
python -m venv venv
source venv/bin/activate
```

Install required libraries
```
pip install pandas matplotlib seaborn
```

Create plots (Make sure results.csv exists)
```
python3 viz.py
```
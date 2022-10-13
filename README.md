# Implementation for the submitted paper "Sample-Then-Optimize Batch Neural Thompson Sampling"

The code here is for the Lunar-Lander task in Sec. 5.3 of the paper (Fig. 3a).
We have made use of the implementation from https://github.com/bobby-he/bayesian-ntk, which is implemented based on the neural-tangents package. For Neural UCB and Neural TS, we have made use of the opensourced code from the Neural TS paper: https://github.com/ZeroWeight/NeuralTS, and we use all default parameters from the code there.


## Requirements:
'pip install -r requirements.txt'

The following commands are also requried for running the Lunar-Lander environment.
'conda install swig'
'pip install box2d-py'

## Instructions to run:
- lunar_gp_bo.py: runs the GP-TS and GP-UCB algorithms.
- lunar_sto_bnts.py: runs our STO-BNTS and STO-BNTS-Linear algorithms

## Analysis and Visualization of results:
- analyze.ipynb

## Decriptions:
- sto_bnts.py, helper_funcs_sto_bnts.py: implementations of our STO-BNTS and STO-BNTS-Linear
- bayesian_optimization_gp.py, helper_funcs_gp_bo.py: implementations of GP-TS and GP-UCB

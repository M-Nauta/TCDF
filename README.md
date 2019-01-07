# TCDF: Temporal Causal Discovery Framework

The Temporal Causal Discovery Framework (TCDF) is a deep learning framework implemented in PyTorch that learns a causal graph structure by discovering causal relationships in observational time series data. TCDF uses Attention-based Convolutional Neural Networks combined with a causal validation step. By interpreting the internal parameters of the convolutional networks, TCDF can also discover the time delay between a cause and the occurrence of its effect. Our framework learns temporal causal graphs, which can include confounders and instantaneous effects.  This broadly applicable framework can be used to gain novel insights into the causal dependencies in a complex system, which is important for reliable predictions, knowledge discovery and data-driven decision making.

Corresponding Paper: ["Causal Discovery with Attention-Based Convolutional Neural Networks"](https://www.mdpi.com/2504-4990/1/1/19).
Please cite this paper when using TCDF.

## Functionality

* Predicts one time series based on other time series and its own historical values using CNNs
* Discovers causal relationships between time series
* Discovers time delay between cause and effect
* Plots temporal causal graph
* Plots predicted time series

## Prerequisites

### General
* Python >= 3.5
* [PyTorch](https://pytorch.org/get-started/locally/) (tested with PyTorch 0.4.1)
* Optional: CUDA (tested with CUDA 9.2) 

### Required Python Packages:
* numpy
* pandas
* random
* heapq
* copy
* os
* sys
* matplotlib
* pylab
* networkx
* argparse

### Data
Required: Dataset(s) containing multiple continuous time series (such as stock markets). 

File format: 
CSV file (comma separated) with header and a column for each time series. 

#### Data provided
The folder 'data' contains two benchmarks:
1. Financial benchmark with stock returns, taken from [S. Kleinberg](http://www.skleinberg.org/data.html) (Finance CPT) and preprocessed
2. Neuroscientific FMRI benchmark with brain networks, taken from [Smith et al.](http://www.fmrib.ox.ac.uk/datasets/netsim/) and preprocessed

Furthermore, there is one small dataset for demonstration purposes (which is a subset of a financial dataset).

### Running

Run `runTCDF.py --data yourdataset.csv` to run TCDF on your own dataset. TCDF will discover causal relationships between time series in the dataset and their time delay. If the ground truth is available, the results of TCDF can be compared with the ground truth for evaluation as follows: `runTCDF.py --ground_truth yourdataset.csv=yourgroundtruth.csv`. Use --help to see all argument options.

To evaluate the predictions made by TCDF, run `evaluate_predictions_TCDF`. Use --help to see all argument options.

Check out the Jupyter Notebook `TCDF Demo` to see an example. 


 

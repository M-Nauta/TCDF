# TCDF: Temporal Causal Discovery Framework

The Temporal Causal Discovery Framework (TCDF) is a deep learning framework implemented in PyTorch. Given multiple time series as input, TCDF discovers **causal relationships** between these time series and outputs a causal graph. It can also predict one time series based on other time series. TCDF uses Attention-based Convolutional Neural Networks combined with a causal validation step. By interpreting the internal parameters of the convolutional networks, TCDF can also discover the **time delay** between a cause and its effect. 

The learnt temporal causal graphs can include confounders and instantaneous effects.  This broadly applicable framework can be used to gain novel insights into the causal dependencies in a complex system, which is important for reliable predictions, knowledge discovery and data-driven decision making. 

Corresponding Paper: ["Causal Discovery with Attention-Based Convolutional Neural Networks"](https://www.mdpi.com/2504-4990/1/1/19).

![Multivariate time series as input, causal graph as output](https://res.mdpi.com/make/make-01-00019/article_deploy/html/images/make-01-00019-g001.png)

## Functionality

* Predicts one time series based on other time series and its own historical values using CNNs
* Discovers causal relationships between time series
* Discovers time delay between cause and effect
* Plots temporal causal graph
* Plots predicted time series

Check out the Jupyter Notebook `TCDF Demo` to see a demonstration of the functionality. 
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
1. Financial benchmark with stock returns, taken from [S. Kleinberg](http://www.skleinberg.org/data.html) (Finance CPT) and preprocessed. Files with 'returns' in the filename are the input datasets, the other files contain the ground truth. 
2. Neuroscientific FMRI benchmark with brain networks, taken from [Smith et al.](http://www.fmrib.ox.ac.uk/datasets/netsim/) and preprocessed. Files with 'timeseries' in the filename are the input datasets, the other files contain the ground truth. 

Furthermore, there is one small dataset for demonstration purposes.

### Running

Check out the Jupyter Notebook `TCDF Demo` to see a demonstration of the functionality. 

Run `runTCDF.py --data yourdataset.csv` to run TCDF on your own dataset(s). TCDF will discover causal relationships between time series in the dataset and their time delay. 

If the ground truth is available, the results of TCDF can be compared with the ground truth for evaluation as follows: `runTCDF.py --ground_truth yourdataset.csv=yourgroundtruth.csv`. Use --help to see all argument options.

To evaluate the predictions made by TCDF, run `evaluate_predictions_TCDF`. Use --help to see all argument options.

_Feel free to improve TCDF. Some [closed issues](https://github.com/M-Nauta/TCDF/issues?q=is%3Aissue+is%3Aclosed) already mention some suggestions._  
 
## Paper

Corresponding Paper (peer-reviewed, open access): ["Causal Discovery with Attention-Based Convolutional Neural Networks"](https://www.mdpi.com/2504-4990/1/1/19). 
Please cite this paper when using TCDF:

```
@article{nauta2019causal,
  title={Causal Discovery with Attention-Based Convolutional Neural Networks},
  author={Nauta, Meike and Bucur, Doina and Seifert, Christin},
  journal={Machine Learning and Knowledge Extraction},
  volume={1},
  number={1},
  pages={312-340},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

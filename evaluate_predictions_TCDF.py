import TCDF
import argparse
import torch
import torch.optim as optim
from model import ADDSTCN
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys

# os.chdir(os.path.dirname(sys.argv[0])) #uncomment this line to run in VSCode

def check_positive(value):
    """Checks if argument is positive integer (larger than zero)."""
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

def check_zero_or_positive(value):
    """Checks if argument is positive integer (larger than or equal to zero)."""
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

class StoreDictKeyPair(argparse.Action):
    """Creates dictionary containing datasets as keys and ground truth files as values."""
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def check_between_zero_and_one(value):
    """Checks if argument is float between zero and 1."""
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
         raise argparse.ArgumentTypeError("%s should be a float between 0 and 1" % value)
    return fvalue

def evaluate_prediction(target, cuda, epochs, kernel_size, layers, 
               loginterval, lr, optimizername, seed, dilation_c, split, file):
    """Runs first part of TCDF to predict one time series and evaluate its accuracy (MASE)."""
    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    X, Y = TCDF.preparedata(file, target)
    X = X.unsqueeze(0).contiguous()
    Y = Y.unsqueeze(2).contiguous()

    timesteps = X.size()[2]
    if timesteps!=Y.size()[1]:
        print("WARNING: Time series do not have the same length.")
    X_train = X[:,:,:int(split*timesteps)]
    Y_train = Y[:,:int(split*timesteps),:]
    X_test = X[:,:,int(split*timesteps):]
    Y_test = Y[:,int(split*timesteps):,:]

    input_channels = X_train.size()[1]
    targetidx = pd.read_csv(file).columns.get_loc(target)
          
    model = ADDSTCN(targetidx, input_channels, levels, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()

    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    for ep in range(1, epochs+1):
        scores, realloss = TCDF.train(ep, X_train, Y_train, model, optimizer,loginterval,epochs)
    realloss = realloss.cpu().data.item()

    model.eval()
    output = model(X_test)
    prediction=output.cpu().detach().numpy()[0,:,0]
    T = output.size()[1]
    total_e = 0.
    for t in range(T):
        real = Y_test[:,t,:]
        predicted = output[:,t,:]
        e = abs(real - predicted)
        total_e+=e
    total_e = total_e.cpu().data.item()
    total = 0.
    for t in range(1,T):
        temp = abs(Y_test[:,t,:] - Y_test[:,t-1,:])
        total+=temp
    denom = (T/float(T-1))*total
    denom = denom.cpu().data.item()

    if denom!=0.:
        MASE = total_e/float(denom)
    else:
        MASE = 0.
    
    return MASE, prediction

def plot_predictions(predictions, file):
    """Plots the predicted values of all time series in the dataset"""
    for c in predictions:
        p = predictions[c]
        plt.plot(p,label=c)
        plt.xlabel('Time')
        plt.ylabel('Predicted value')
        plt.title('Dataset %s'%file)
        plt.legend()

    plt.show()

def evaluate(datafile):
    """Collects the predictions of all time series in a dataset and returns overall results."""
    stringdatafile = str(datafile)
    if '/' in stringdatafile:
        stringdatafile = str(datafile).rsplit('/', 1)[1]
    df_data = pd.read_csv(datafile)
    columns = list(df_data)

    MASEs = []
    predictions = dict()
    for c in columns:
        MASE, prediction = evaluate_prediction(c, cuda=cuda, epochs=nrepochs, 
        kernel_size=kernel_size, layers=levels, loginterval=loginterval, 
        lr=learningrate, optimizername=optimizername,
        seed=seed, dilation_c=dilation_c, split=split, file=datafile)
        predictions[c]= prediction
        MASEs.append(MASE)
        allres.append(MASE)
    avg = np.mean(MASEs)
    std = np.std(MASEs)
    
    return allres, avg, std, predictions
parser = argparse.ArgumentParser(description='TCDF: Temporal Causal Discovery Framework')

parser.add_argument('--cuda', action="store_true", default=False, help='Use CUDA (GPU) (default: False)')
parser.add_argument('--epochs', type=check_positive, default=1000, help='Number of epochs (default: 1000)')
parser.add_argument('--kernel_size', type=check_positive, default=4, help='Size of sliding kernel (default: 4)')
parser.add_argument('--hidden_layers', type=check_zero_or_positive, default=0, help='Number of hidden layers in the depthwise convolution (default: 0)') 
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop'], help='Optimizer to use: Adam or RMSprop (default: Adam)')
parser.add_argument('--log_interval', type=check_positive, default=500, help='Epoch interval to report loss (default: 500)')
parser.add_argument('--seed', type=check_positive, default=1111, help='Random seed (default: 1111)')
parser.add_argument('--dilation_coefficient', type=check_positive, default=4, help='Dilation coefficient, recommended to be equal to kernel size (default: 4)')
parser.add_argument('--plot', action="store_true", default=False, help='Plot predicted time series (default: False)')
parser.add_argument('--train_test_split', type=check_between_zero_and_one, default=0.8, help="Portion of dataset to use for training (default 0.8)")
parser.add_argument('--data', nargs='+', required=True, help='(Path to) Dataset(s) to predict by TCDF containing multiple time series. Required file format: csv with a column (incl. header) for each time series')

args = parser.parse_args()

print("Arguments:", args)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, you should probably run with --cuda to speed up training.")
if args.kernel_size != args.dilation_coefficient:
    print("WARNING: The dilation coefficient is not equal to the kernel size. Multiple paths can lead to the same delays. Set kernel_size equal to dilation_c to have exaxtly one path for each delay.")

kernel_size = args.kernel_size
levels = args.hidden_layers+1
nrepochs = args.epochs
learningrate = args.learning_rate
optimizername = args.optimizer
dilation_c = args.dilation_coefficient
loginterval = args.log_interval
seed=args.seed
cuda=args.cuda
split=args.train_test_split
plot = args.plot
datasets = args.data

evalresults = dict()
allres = []
for datafile in datasets:
    allres,avg,std,predictions = evaluate(datafile)
    evalresults[datafile]=(avg, std)
    print("\nMean Absolute Scaled Error (MASE) averaged over all time series in", datafile,":",evalresults[datafile][0],"with standard deviation",evalresults[datafile][1])
    if plot:
        plot_predictions(predictions,datafile)

if len(datasets)>1:
    overallavg = np.mean(allres)
    overallstd = np.std(allres)
    print("=========================Overall Evaluation====================================")
    print("Average MASE over all datasets: ", overallavg)
    print("Standard Deviation MASE over all datasets: ", overallstd)


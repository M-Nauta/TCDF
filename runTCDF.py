import TCDF
import argparse
import torch
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

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships."""
    gtdata = pd.read_csv(gtfile, header=None)

    readgt=dict()
    effects = gtdata[1]
    causes = gtdata[0]
    delays = gtdata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k]=[]
    for i in range(len(effects)):
        key=effects[i]
        value=causes[i]
        readgt[key].append(value)
        pairdelays[(key, value)]=delays[i]
        gtnrrelations+=1
    
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                #find extended delay
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    delay=0
                    for i in range(len(p)-1):
                        delay+=pairdelays[(p[i+1], p[i])]
                    extendedgtdelays[(effect, cause)].append(delay)

    return extendedgtdelays, readgt, extendedreadgt

def evaluate(gtfile, validatedcauses, columns):
    """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
          
    print("Total False Positives': ", FP)
    print("Total True Positives': ", TP)
    print("Total False Negatives: ", FN)
    print("Total Direct False Positives: ", FPdirect)
    print("Total Direct True Positives: ", TPdirect)
    print("TPs': ", TPs)
    print("FPs': ", FPs)
    print("TPs direct: ", TPsdirect)
    print("FPs direct: ", FPsdirect)
    print("FNs: ", FNs)
    precision = recall = 0.

    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    print("Precision': ", precision)
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    print("Recall': ", recall)
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    print("F1' score: ", F1,"(includes direct and indirect causal relationships)")

    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    print("Precision: ", precision)
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    print("Recall: ", recall)
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    print("F1 score: ", F1direct,"(includes only direct causal relationships)")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct

def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
                
            else:
                next
           
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)


def runTCDF(datafile):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)

    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()

    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(c, cuda=cuda, epochs=nrepochs, 
        kernel_size=kernel_size, layers=levels, log_interval=loginterval, 
        lr=learningrate, optimizername=optimizername,
        seed=seed, dilation_c=dilation_c, significance=significance, file=datafile)

        allscores[idx]=scores
        allcauses[idx]=causes
        alldelays.update(causeswithdelay)
        allreallosses[idx]=realloss

    return allcauses, alldelays, allreallosses, allscores, columns

def plotgraph(stringdatafile,alldelays,columns):
    """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in alldelays:
        p1,p2 = pair
        nodepair = (columns[p2], columns[p1])

        G.add_edges_from([nodepair],weight=alldelays[pair])
    
    edge_labels=dict([((u,v,),d['weight'])
                    for u,v,d in G.edges(data=True)])
    
    pos=nx.circular_layout(G)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos, node_color = 'white', edge_color='black',node_size=1000,with_labels = True)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000") 

    pylab.show()

def main(datafiles, evaluation):
    if evaluation:
        totalF1direct = [] #contains F1-scores of all datasets
        totalF1 = [] #contains F1'-scores of all datasets

        receptivefield=1
        for l in range(0, levels):
            receptivefield+=(kernel_size-1) * dilation_c**(l)

    for datafile in datafiles.keys(): 
        stringdatafile = str(datafile)
        if '/' in stringdatafile:
            stringdatafile = str(datafile).rsplit('/', 1)[1]
        
        print("\n Dataset: ", stringdatafile)

        # run TCDF
        allcauses, alldelays, allreallosses, allscores, columns = runTCDF(datafile) #results of TCDF containing indices of causes and effects

        print("\n===================Results for", stringdatafile,"==================================")
        for pair in alldelays:
            print(columns[pair[1]], "causes", columns[pair[0]],"with a delay of",alldelays[pair],"time steps.")

        

        if evaluation:
            # evaluate TCDF by comparing discovered causes with ground truth
            print("\n===================Evaluation for", stringdatafile,"===============================")
            FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(datafiles[datafile], allcauses, columns)
            totalF1.append(F1)
            totalF1direct.append(F1direct)

            # evaluate delay discovery
            extendeddelays, readgt, extendedreadgt = getextendeddelays(datafiles[datafile], columns)
            percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, receptivefield)*100
            print("Percentage of delays that are correctly discovered: ", percentagecorrect,"%")
            
        print("==================================================================================")
        
        if args.plot:
            plotgraph(stringdatafile, alldelays, columns)

    # In case of multiple datasets, calculate average F1-score over all datasets and standard deviation
    if len(datafiles.keys())>1 and evaluation:  
        print("\nOverall Evaluation: \n")      
        print("F1' scores: ")
        for f in totalF1:
            print(f)
        print("Average F1': ", np.mean(totalF1))
        print("Standard Deviation F1': ", np.std(totalF1),"\n")
        print("F1 scores: ")
        for f in totalF1direct:
            print(f)
        print("Average F1: ", np.mean(totalF1direct))
        print("Standard Deviation F1: ", np.std(totalF1direct))




parser = argparse.ArgumentParser(description='TCDF: Temporal Causal Discovery Framework')

parser.add_argument('--cuda', action="store_true", default=False, help='Use CUDA (GPU) (default: False)')
parser.add_argument('--epochs', type=check_positive, default=1000, help='Number of epochs (default: 1000)')
parser.add_argument('--kernel_size', type=check_positive, default=4, help='Size of kernel, i.e. window size. Maximum delay to be found is kernel size - 1. Recommended to be equal to dilation coeffient (default: 4)')
parser.add_argument('--hidden_layers', type=check_zero_or_positive, default=0, help='Number of hidden layers in the depthwise convolution (default: 0)') 
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop'], help='Optimizer to use (default: Adam)')
parser.add_argument('--log_interval', type=check_positive, default=500, help='Epoch interval to report loss (default: 500)')
parser.add_argument('--seed', type=check_positive, default=1111, help='Random seed (default: 1111)')
parser.add_argument('--dilation_coefficient', type=check_positive, default=4, help='Dilation coefficient, recommended to be equal to kernel size (default: 4)')
parser.add_argument('--significance', type=float, default=0.8, help="Significance number stating when an increase in loss is significant enough to label a potential cause as true (validated) cause. See paper for more details (default: 0.8)")
parser.add_argument('--plot', action="store_true", default=False, help='Show causal graph (default: False)')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--ground_truth',action=StoreDictKeyPair, help='Provide dataset(s) and the ground truth(s) to evaluate the results of TCDF. Argument format: DataFile1=GroundtruthFile1,Key2=Value2,... with a key for each dataset containing multivariate time series (required file format: csv, a column with header for each time series) and a value for the corresponding ground truth (required file format: csv, no header, index of cause in first column, index of effect in second column, time delay between cause and effect in third column)')
group.add_argument('--data', nargs='+', help='(Path to) one or more datasets to analyse by TCDF containing multiple time series. Required file format: csv with a column (incl. header) for each time series')

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
significance=args.significance

if args.ground_truth is not None:
    datafiles = args.ground_truth
    main(datafiles, evaluation=True)

else:
    datafiles = dict()
    for dataset in args.data:
        datafiles[dataset]=""
    main(datafiles, evaluation=False)

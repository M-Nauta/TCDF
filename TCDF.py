import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ADDSTCN
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys

def preparedata(file, target):
    """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
    df_data = pd.read_csv(file)
    df_y = df_data.copy(deep=True)[[target]]
    df_x = df_data.copy(deep=True)
    df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
    df_yshift[target]=df_yshift[target].fillna(0.)
    df_x[target] = df_yshift
    data_x = df_x.values.astype('float32').transpose()    
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    x, y = Variable(data_x), Variable(data_y)
    return x, y


def train(epoch, traindata, traintarget, modelname, optimizer,log_interval,epochs):
    """Trains model by performing one epoch and returns attention scores and loss."""

    modelname.train()
    x, y = traindata[0:1], traintarget[0:1]
        
    optimizer.zero_grad()
    epochpercentage = (epoch/float(epochs))*100
    output = modelname(x)

    attentionscores = modelname.fs_attention
    
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()

    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

    return attentionscores.data, loss

def findcauses(target, cuda, epochs, kernel_size, layers, 
               log_interval, lr, optimizername, seed, dilation_c, significance, file):
    """Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers the corresponding time delays"""

    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    X_train, Y_train = preparedata(file, target)
    X_train = X_train.unsqueeze(0).contiguous()
    Y_train = Y_train.unsqueeze(2).contiguous()

    input_channels = X_train.size()[1]
       
    targetidx = pd.read_csv(file).columns.get_loc(target)
          
    model = ADDSTCN(targetidx, input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
    firstloss = firstloss.cpu().data.item()
    for ep in range(2, epochs+1):
        scores, realloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
    realloss = realloss.cpu().data.item()
    
    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
    
    #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i]>1.:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): #gap should be in first half
                if index>0:
                    ind=index #gap should have index > 0, except if second score <1
                    break
        if ind<0:
            ind = 0
                
        potentials = indices[:ind+1].tolist()
    print("Potential causes: ", potentials)
    validated = copy.deepcopy(potentials)
    
    #Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in potentials:
        random.seed(seed)
        X_test2 = X_train.clone().cpu().numpy()
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        if cuda:
            shuffled=shuffled.cuda()
        model.eval()
        output = model(shuffled)
        testloss = F.mse_loss(output, Y_train)
        testloss = testloss.cpu().data.item()
        
        diff = firstloss-realloss
        testdiff = firstloss-testloss

        if testdiff>(diff*significance): 
            validated.remove(idx) 
    
 
    weights = []
    
    #Discover time delay between cause and effect by interpreting kernel weights
    for layer in range(layers):
        weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], model.dwn.network[layer].net[0].weight.size()[2])
        weights.append(weight)

    causeswithdelay = dict()    
    for v in validated: 
        totaldelay=0    
        for k in range(len(weights)):
            w=weights[k]
            row = w[v]
            twolargest = heapq.nlargest(2, row)
            m = twolargest[0]
            m2 = twolargest[1]
            if m > m2:
                index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
            else:
                #take first filter
                index_max=0
            delay = index_max *(dilation_c**k)
            totaldelay+=delay
        if targetidx != v:
            causeswithdelay[(targetidx, v)]=totaldelay
        else:
            causeswithdelay[(targetidx, v)]=totaldelay+1
    print("Validated causes: ", validated)
    
    return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()






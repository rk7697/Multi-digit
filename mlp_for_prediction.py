print("test")
print("checker")
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.utils.data import Sampler
import random
import statistics
import time
from torch.nn import init
from matplotlib.ticker import FormatStrFormatter
import mpld3
class net(nn.Module):
    def __init__(self,weightinit,linear=True):
        super().__init__()
        self.activation=nn.ReLU()
        self.linear=linear
        ##Layer instantiation if mlp
        if linear:
            ###Linear init 1
            self.linear1=nn.Linear(28*28,400)
            self.linear2=nn.Linear(400,200)
            self.linear3=nn.Linear(200,100)
            self.linear4=nn.Linear(100,50)
            self.linear5=nn.Linear(50,25)
            self.linear6=nn.Linear(25,10)

    ###Forward pass
    def forward(self,x):
        self.recentoutputs=[]
        if self.linear:
            linearlayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Linear)))]
            x=torch.flatten(x,start_dim=1)
            layerindex=0
            x.retain_grad()
            for layer in linearlayers:
                layerindex+=1
                x=layer(x)
                if(layer != linearlayers[-1]):
                    x=self.activation(x)
            return x
        else:
            cnnsandmaxpoollayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Conv2d)) or isinstance(layer,(nn.MaxPool2d)))]
            linearlayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Linear)))]
            layerindex=0
            self.forwardpassvariances[0].append(torch.var(x).item())
            for layer in (cnnsandmaxpoollayers):
                if(isinstance(layer,nn.Conv2d)):
                    layerindex+=1
                    x=layer(x)
                    x=self.activation(x)
            x=torch.flatten(x,start_dim=1)
            for layer in linearlayers:
                layerindex+=1
                x=layer(x)
                self.recentoutputs.append(x)
                if(layer != linearlayers[-1]):
                    x=self.activation(x)
            return x     
    ###
dataset="MNIST_new"
rootdir=os.path.join("/Users/rileyknutson/desktop/",dataset)
optimizer=nn.
class MNISTdata(Dataset):
    def __init__(self, rootdir, data):
        self.data=data
        self.rootdir=rootdir
        self.dir=os.path.join(rootdir,data+".csv")
        self.imgs=pd.read_csv(self.dir)
        if (data=="train"):
            self.labels=pd.read_csv(self.dir)['label']
    def __getitem__(self, idx):
        # if(self.data=="train"):
        imgdata=self.imgs.iloc[idx].values[1:]
        imgdata=imgdata.reshape(28,28)
        pixels=torch.tensor(imgdata,requires_grad=True,dtype=torch.float32)
        pixels=pixels/255.0
        pixels=pixels.unsqueeze(dim=0)
        label=torch.tensor(self.labels[idx])
        return label, pixels
    def __len__(self):
        return len(self.imgs)
class CustomSampler(Sampler):
    def __init__(self, permutation):
        self.permutation = permutation

    def __iter__(self):
        return iter(self.permutation)

    def __len__(self):
        return len(self.permutation)
trainData=MNISTdata(rootdir,"train")


def train(net,optimizer,epochs):
    lossfunc=nn.CrossEntropyLoss()
    errors=[]
    epochcount=[]
    parametervariances=[[] for _ in range(model.total_layers)]
    forwardpassvariances=[]
    backwardpassvariances=[[] for _ in range(model.total_layers+1)]
    #Add 1 to total_layers because we will also track the loss gradient for the very first input
    if("sgd" in optimizer):
        optim=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    else:
        optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0000001)
    for epoch in range(epochs):
        traindataloader=DataLoader(trainData,batch_size=32,shuffle=True)
        runningerror=0.0
        i=0
        z=0
        for labels, imgs in traindataloader:
            z+=1
            optim.zero_grad()
            logits=model.forward(imgs)
            loss=lossfunc(logits,labels)
            loss.backward()
            if(name=="kaiming_new"):
                model.normalize_weights()

            optim.step()
            runningerror+=loss.detach()

            recentparametergradientvariances=mostrecentgradientvariances(model)
            for x in range(len(recentparametergradientvariances)):
                parametervariances[x].append(recentparametergradientvariances[x])

            recentoutputgradients=[output.grad for output in model.recentoutputs]
            ###DO NOT USE I!!
            for x in range(len(recentoutputgradients)):
                backwardpassvariances[x].append(torch.var(recentoutputgradients[x]).item())
            ###
            if(i>0 and i%(10)==0):
                avgerror=runningerror/i
                errors.append(avgerror)
                

                percent=epoch+i/len(traindataloader)
                epochcount.append(percent)
                if(i%100==0):
                    print("error:{:<10} epoch:{:<2} percent:{:<7.2f}% round:{:<5}".format(avgerror, epoch, percent, r))
                    print("------{}-----".format(name))
            i+=1
    forwardpassvariances=model.forwardpassvariances
    return epochcount,errors,parametervariances,forwardpassvariances, backwardpassvariances, optimizer    

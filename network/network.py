import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
from .dataset import f
f()
exit()


exit()
from dir_path.dataset import f
exit()
print(dir_path)
exit()
from dataset import MNISTdata



class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation=nn.ReLU()
    ###Forward pass
    def forward(self,x):
        if self.linear:
            linearlayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Linear)))]
            x=torch.flatten(x,start_dim=1)
            layerindex=0
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
optimizer=torch.optim.Adam



trainData=MNISTdata(rootdir,"train")

torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0000001)

def train(net,optimizer,epochs):
    lossfunc=nn.CrossEntropyLoss()
    errors=[]
    epochcount=[]
    for epoch in range(epochs):
        traindataloader=DataLoader(trainData,batch_size=32,shuffle=True)
        runningerror=0.0
        i=0
        for labels, imgs in traindataloader:
            optim.zero_grad()
            logits=net.forward(imgs)
            loss=lossfunc(logits,labels)
            loss.backward()
            optim.step()
            runningerror+=loss
            if(i>0 and i%(10)==0):
                avgerror=runningerror/i
                errors.append(avgerror)
                percent=epoch+i/len(traindataloader)
                epochcount.append(percent)
                if(i%100==0):
                    print("error:{:<10} epoch:{:<2} percent:{:<7.2f}%".format(avgerror, epoch, percent))
            i+=1
    return epochcount,errors,  

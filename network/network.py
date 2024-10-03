import torchvision.transforms as transforms
from .dataloaders import (
    train_dataloader,
    test_dataloader
)
import matplotlib.pyplot as plt

to_pil = transforms.ToPILImage()

for imgs, labels in train_dataloader:
    print(imgs.shape)
    imgs=imgs[0]
    img=to_pil(imgs)
    plt.imshow(img, cmap='gray')
    plt.show()

    
    exit()
    print(imgs.shape)
    exit()
# class net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.activation=nn.ReLU()
#     def forward(self,x):
#             cnnsandmaxpoollayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Conv2d)) or isinstance(layer,(nn.MaxPool2d)))]
#             linearlayers=[layer for layer in list(self.children()) if (isinstance(layer,(nn.Linear)))]
#             layerindex=0
#             for layer in (cnnsandmaxpoollayers):
#                 if(isinstance(layer,nn.Conv2d)):
#                     layerindex+=1
#                     x=layer(x)
#                     x=self.activation(x)
#             x=torch.flatten(x,start_dim=1)
#             for layer in linearlayers:
#                 layerindex+=1
#                 x=layer(x)
#                 self.recentoutputs.append(x)
#                 if(layer != linearlayers[-1]):
#                     x=self.activation(x)
#             return x     
###

# optimizer=torch.optim.Adam



# trainData=MNISTdata(rootdir,"train")

# torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0000001)

# def train(net,optimizer,epochs):
#     lossfunc=nn.CrossEntropyLoss()
#     errors=[]
#     epochcount=[]
#     for epoch in range(epochs):
#         traindataloader=DataLoader(trainData,batch_size=32,shuffle=True)
#         runningerror=0.0
#         i=0
#         for labels, imgs in traindataloader:
#             optim.zero_grad()
#             logits=net.forward(imgs)
#             loss=lossfunc(logits,labels)
#             loss.backward()
#             optim.step()
#             runningerror+=loss
#             if(i>0 and i%(10)==0):
#                 avgerror=runningerror/i
#                 errors.append(avgerror)
#                 percent=epoch+i/len(traindataloader)
#                 epochcount.append(percent)
#                 if(i%100==0):
#                     print("error:{:<10} epoch:{:<2} percent:{:<7.2f}%".format(avgerror, epoch, percent))
#             i+=1
#     return epochcount,errors,  

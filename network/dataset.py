from torchvision import datasets
from torchvision import transforms

#Calculate padding to pad reach 500x500 from 28x28
padding=(500-28)//2

transform = transforms.Compose([
    transforms.Pad(padding=padding,fill=255), #Pad 
    transforms.ToTensor()
])

dataset = datasets.MNIST("./data",train=True,download=True,transform=transform)
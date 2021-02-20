import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch._C import has_cuda
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms

train = pd.read_csv("./sample_data/mnist_train_small.csv",dtype = np.float32)
train.rename(columns={'6':'label'},inplace=True)
test = pd.read_csv("./sample_data/mnist_test.csv",dtype = np.float32)
test.rename(columns={'7':'label'},inplace=True)

train = train.query("label in [7.0, 8.0]").head(400)

test = test.query("label in [2.0, 7.0 ,8.0]").head(600)

train = train.iloc[:,1:].values.astype('float32')
test = test.iloc[:,1:].values.astype('float32')

train = train.reshape(train.shape[0],28,28)
test = test.reshape(test.shape[0],28,28)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(330+(i+1))
    plt.imshow(train[i],cmap=plt.get_cmap('gray'))
#plt.show()

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        def CBA(in_channel,out_channel,kernel_size=4,stride=2,padding=1,activation=nn.ReLU(inplace=True),bn=True):
            seq = []
            seq += [nn.ConvTranspose2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding)]

            if bn is True:
                seq += [nn.BatchNorm2d(out_channel)]
            seq +=  [activation]

            return nn.Sequential(*seq)
        
        seq = []
        seq += [CBA(20,64*8,stride=1,padding=0)]
        seq += [CBA(64*8,64*4)]
        seq += [CBA(64*4,64*2)]
        seq += [CBA(64*2,64)]
        seq += [CBA(64,1,activation=nn.Tanh(),bn=False)]

        self.generator_network = nn.Sequential(*seq)

    def forward(self,z):
        out=self.generator_network(z)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        def CBA(in_channel,out_channel,kernel_size=4, stride=2, padding=1,activation=nn.LeakyReLU(0.1,inplace=True)):
            seq = []
            seq += [nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding)]
            seq += [nn.BatchNorm2d(out_channel)]
            seq += [activation]

        seq = []
        seq += [CBA(1,64)]
        seq += [CBA(64,64*2)]
        seq += [CBA(64*2,64*4)]
        seq += [CBA(64*4,64*8)]
        
        self.feature_network = nn.Conv2d(64*8,1,kernel_size=4,stride=1)

    
    def forward(self,x):
        out = self.feature_network(x)

        feature = out
        feature = feature.view(feature.size(0),-1)

        out = self.critic_network(out)
        return out,feature

class image_data_set(Dataset):
    def __init__(self,data):
        self.images = data[:,:,:,None]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64,interpolation=Image.BICUBIC),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        return self.transform(self.images[idx])
    
train_set = image_data_set(train)
train_loader = DataLoader(train_set,batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)

G.train()
D.train()

optimizerG = torch.optim.Adam(G.parameters(),lr=0.0001,betas=(0.0,0.9))
optimizerD = torch.optim.Adam(D.parameters(),lr=0.0004,betas=(0.0,0.9))

criterion = nn.BCEWithLogitsLoss(reduction='mean')

for epoch in range(300):
    log_g_loss , log_d_loss = 0.0,0.0
    for images in train_loader:
        images = images.to(device)
        
        label_real = torch.full((images.size(0),),1.0).to(device)
        label_fake = torch.full((images.size(0),),0.0).to(device)

        z = torch.randn(images.size(0),20).to(device).view(images.size(0),20,1,1,).to(device)
        fake_images = G(z)

        d_out_real , _ = D(images)
        d_out_fake , _ = D(fake_images)

        d_loss_real = criterion(d_out_real.view(-1),label_real)
        d_loss_fake = criterion(d_out_fake.view(-1),label_fake)

        d_loss = d_loss_real + d_loss_fake

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        z = torch.randn(images.size(0),20).to(device).view(images.size(0),20,1,1).to(device)
        fake_images = G(z)

        d_out_fake,_ = D(fake_images)

        g_loss = criterion(d_out_fake.view(-1),label_real)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        log_d_loss += d_loss.item()
        log_g_loss += g_loss.item()

    print(f'epoch {epoch}, D_Loss{log_d_loss/128:.4f} , G_Loss:{log_g_loss/128:.4f}')

z = torch.randn(8,20).to(device).view(8,20,1,1).to(device)
fake_image = G(z)

fig = plt.figure(figsize=(15,3))
for i in range(0,5):
    plt.subplot(1,5,i+1)
    plt.imshow(fake_images[i][0].cpu().etach().numpy(),'gray')
plt.show()

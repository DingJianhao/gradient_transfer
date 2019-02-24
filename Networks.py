import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MLP_Embedding(nn.Module):
    def __init__(self):
        super(MLP_Embedding, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(28*28, 256),
                                # nn.PReLU(),
                                nn.Linear(256, 256)
                                # nn.PReLU(),
                                )
        self.fc2 = nn.Sequential(nn.Linear(256,16))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        output = self.fc2(output)
        return output


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class Generator(nn.Module):
    def __init__(self,dim):
        super(Generator, self).__init__()
        self.dim = dim
        self.size = self.dim[0]*self.dim[1]
        self.fc = nn.Sequential(
            nn.Linear(self.size , 100),
            nn.Linear(100, self.size)
        )

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x.view(x.size(0),self.dim[0],self.dim[1])


class Discriminator(nn.Module):
    def __init__(self,dim):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.size = self.dim[0]*self.dim[1]
        self.fc = nn.Sequential(
            nn.Linear(self.size , 100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
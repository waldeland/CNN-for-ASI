import torch
from torch import nn


class TextureNet(nn.Module):
    def __init__(self, use_gpu=True):
        self.use_gpu=use_gpu
        super(TextureNet,self).__init__()

        # Network definition
        self.net = nn.Sequential(
            nn.Conv3d(1,50,5,4,padding=2), #Parameters  #in_channels, #out_channels, filter_size, stride (downsampling factor)
            nn.BatchNorm3d(50),
            #nn.Dropout3d() #Droput can be added like this ...
            nn.ReLU(),

            nn.Conv3d(50,50,3,2,padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),

            nn.Conv3d(50,50,3,2,padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),

            nn.Conv3d(50,50,3,2,padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),

            nn.Conv3d(50,50,3,3,padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),

            nn.Conv3d(50,2,1,1), #This is the equivalent of a fully connected layer since input has width/height/depth = 1
            nn.ReLU(),

        )
        #The filter weights are by default initialized by random

    #Is called to compute network output
    def forward(self,x):
        return self.net(x)



    def classify(self,x):
        x = self.net(x)
        _, class_no = torch.max(x, 1, keepdim=True)
        return class_no


    # Functions to get output from intermediate feature layers
    def f1(self, x,):
        return self.getFeatures( x, 0)
    def f2(self, x,):
        return self.getFeatures( x, 1)
    def f3(self, x,):
        return self.getFeatures( x, 2)
    def f4(self, x,):
        return self.getFeatures( x, 3)
    def f5(self, x,):
        return self.getFeatures( x, 4)


    def getFeatures(self, x, layer_no):
        layer_indexes = [0, 3, 6, 9, 12]

        #Make new network that has the layers up to the requested output
        tmp_net = nn.Sequential()
        layers = list(self.net.children())[0:layer_indexes[layer_no]+1]
        for i in range(len(layers)):
            tmp_net.add_module(str(i),layers[i])
        if self.use_gpu: tmp_net.cuda()
        return tmp_net(x)



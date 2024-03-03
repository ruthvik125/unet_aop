from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
    def __init__(self,inChannels,outChannels):
        super().__init__()

        self.conv1 = Conv2d(inChannels,outChannels,3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels,outChannels,3)

    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))



class Encoder(Module):
    def __init__(self,channels=(3,16,32,64)):
        #channels tuple, where the first el is the input channels
        super().__init__()

        self.encBlocks = ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
        self.pool = MaxPool2d(2)

    def forward(self,x):
        Blockoutputs = []

        for block in self.encBlocks:
            x= block(x)
            Blockoutputs.append(x)
            x =self.pool(x)
        return Blockoutputs


class Decoder(Module):
    def __init__(self,channels=(64,32,16)):
        super().__init__()

        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i],channels[i+1],2,2) for i in range(len(channels)-1)])
        self.decBlocks = ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
    
    def forward(self,x,encFeatures):

        for i in range(len(self.channels)-1):
            x= self.upconvs[i](x)
            encF = self.crop(encFeatures[i],x)
            x = torch.cat([x,encF],dim=1)
            x = self.decBlocks[i](x)

        return x

    
    def crop(self,encFeature,x):
        (_,_,H,W)=x.shape
        encFeature= CenterCrop([H,W])(encFeature)

        return encFeature



class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64),decChannels=(64, 32, 16),nbClasses=3, retainDim=True,outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
           
		    
    def forward(self, x):
		# grab the features from the encoder
        encFeatures = self.encoder(x)
		    # pass the encoder features through decoder making sure that
		    # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
		encFeatures[::-1][1:])
		    # pass the decoder features through the regression head to
		    # obtain the segmentation mask
        map = self.head(decFeatures)
		    # check to see if we are retaining the original output
		    # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
		    # return the segmentation map
        return map
		
		
		
		
		
		  
		    
            
		   
		    
		



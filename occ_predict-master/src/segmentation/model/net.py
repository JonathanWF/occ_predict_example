"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), 
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
    def forward(self, s1, s2):
        s1 = self.up(s1)
        s = torch.cat([s1,s2], dim = 1)
        s = self.conv(s)
        return s     
        
class Unet(nn.Module):
    def __init__(self, params):
        super(Unet, self).__init__()
        self.num_channels = params.num_channels
        self.num_classes = params.num_classes
        self.down1 = nn.Sequential(
            nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(inplace=True)
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels*2),
            nn.ReLU(inplace=True)
            )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels*4),
            nn.ReLU(inplace=True)
            )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.num_channels*4, self.num_channels*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels*8),
            nn.ReLU(inplace=True)
            )

	
        self.up4 = up(self.num_channels*16, self.num_channels*8)
        self.up3 = up(self.num_channels*12, self.num_channels*4)
        self.up2 = up(self.num_channels*6, self.num_channels*2)
        self.up1 = up(self.num_channels*3, self.num_channels*1)
        
        self.out = nn.Conv2d(self.num_channels, self.num_classes, 1)
    def forward(self, s):
        # s1 is num_channels, result of down 1
        s1 = self.down1(s)
        s = F.max_pool2d(s1, 2)
        #print('down1:')
        #print('s1: {}, s: {}'.format(s1.shape, s.shape))
        
        # s2 is num_channels*2, result of down 2
        s2 = self.down2(s)
        s = F.max_pool2d(s2, 2)
        #print('down2:')
        #print('s2: {}, s: {}'.format(s2.shape, s.shape))

        # s3 is num_channels*4, result of down 3
        s3 = self.down3(s)
        s = F.max_pool2d(s3, 2)
        #print('down3:')
        #print('s3: {}, s: {}'.format(s3.shape, s.shape))
        
        # s4 is num_channels*8, result of down 4
        s4 = self.down4(s)
        s = F.max_pool2d(s4, 2)
        #print('down4:')
        #print('s4: {}, s: {}'.format(s4.shape, s.shape))

        # s+s4=num_channels*16, s is num_channels*8
        s = self.up4(s, s4)
        #print('up4:')
        #print('s: {}'.format(s.shape))
        
        # s+s3=num_channels*12, s is num_channels*6
        s = self.up3(s, s3)
        #print('up3:')
        #print('s: {}'.format(s.shape))

        # s+s2=num_channels*8, s is num_channels*4
        s = self.up2(s, s2)
        #print('up2:')
        #print('s: {}'.format(s.shape))
        
        # s+s1=num_channels*5, s is num_channels
        s = self.up1(s, s1)
        #print('up1:')
        #print('s: {}'.format(s.shape))
        
        # s is num_classes
        s = self.out(s)
        return s

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)



# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

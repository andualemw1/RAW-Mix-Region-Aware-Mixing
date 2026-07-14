
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import sys
# Add the parent directory of 'yolov5' to the Python path
sys.path.append('/home/andualemwelabo.tulu/expt_01/yolov5')
from region_aware.functions import ReverseLayerF

''' https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py'''

'''https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py'''

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Define a simplified domain classifier with fewer layers
        self.domain_classifier = nn.Sequential(
            nn.Linear(512 * 20 * 20, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(p=0.5),  # Dropout with a probability of 0.5
            # nn.GroupNorm(num_groups=16, num_channels=128),  # Group Normalization
            nn.LayerNorm(128),  # Layer Normalization
            nn.ReLU(True),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=-1)  
        )
        
    def forward(self, input, alpha):
        input = input.view(-1, 512 * 20 * 20)
        input = ReverseLayerF.apply(input, alpha)        
        output = self.domain_classifier(input)

        return output


domain_classifier = Discriminator()


import torch
import torch.nn as nn


from .networks import MultiscaleDiscriminator
from torch.cuda.amp import autocast as autocast



class Feature2Face_D(nn.Module):
    def __init__(self, opt):
        super(Feature2Face_D, self).__init__()
        # initialize
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.tD = opt.n_frames_D  
        self.output_nc = opt.output_nc        

        # define networks                    
        self.netD = MultiscaleDiscriminator(23 + 3, opt.ndf, opt.n_layers_D, opt.num_D, not opt.no_ganFeat) 
                    
        print('---------- Discriminator networks initialized -------------') 
        print('-----------------------------------------------------------')
       
    #@autocast()    
    def forward(self, input):
        if self.opt.fp16:
            with autocast():
                pred = self.netD(input)
        else:
            pred = self.netD(input)
        
        return pred
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    

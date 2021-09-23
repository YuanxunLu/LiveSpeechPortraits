import torch.nn as nn

from .networks import Feature2FaceGenerator_Unet, Feature2FaceGenerator_normal, Feature2FaceGenerator_large

from torch.cuda.amp import autocast as autocast


class Feature2Face_G(nn.Module):
    def __init__(self, opt):
        super(Feature2Face_G, self).__init__()
        # initialize
        self.opt = opt
        self.isTrain = opt.isTrain
        # define net G  
        
        if opt.size == 'small':
            self.netG = Feature2FaceGenerator_Unet(input_nc=23, output_nc=3, num_downs=opt.n_downsample_G, ngf=opt.ngf) 
        elif opt.size == 'normal':
            self.netG = Feature2FaceGenerator_normal(input_nc=13, output_nc=3, num_downs=opt.n_downsample_G, ngf=opt.ngf)
        elif opt.size == 'large':
            self.netG = Feature2FaceGenerator_large(input_nc=13, output_nc=3, num_downs=opt.n_downsample_G, ngf=opt.ngf)
            
        print('---------- Generator networks initialized -------------') 
        print('-------------------------------------------------------')
    

    def forward(self, input):
        if self.opt.fp16:
            with autocast():
                fake_pred = self.netG(input)
        else:
            fake_pred = self.netG(input)
                      
        return fake_pred

    











import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from . import networks
from . import feature2face_G
from .base_model import BaseModel
from .losses import GANLoss, MaskedL1Loss, VGGLoss



class Feature2FaceModel(BaseModel):          
    def __init__(self, opt):
        """Initialize the Feature2Face class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['Feature2Face_G']
        self.Feature2Face_G = networks.init_net(feature2face_G.Feature2Face_G(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        if self.isTrain:
            if not opt.no_discriminator:
                self.model_names += ['Feature2Face_D']
                from . import feature2face_D
                self.Feature2Face_D = networks.init_net(feature2face_D.Feature2Face_D(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)


        # define only during training time
        if self.isTrain:
            # define losses names
            self.loss_names_G = ['L1', 'VGG', 'Style', 'loss_G_GAN', 'loss_G_FM']    
            # criterion
            self.criterionMaskL1 = MaskedL1Loss().cuda()
            self.criterionL1 = nn.L1Loss().cuda()
            self.criterionVGG = VGGLoss.cuda()
            self.criterionFlow = nn.L1Loss().cuda()
            
            # initialize optimizer G 
            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr   
            self.optimizer_G = torch.optim.Adam([{'params': self.Feature2Face_G.module.parameters(),
                                                  'initial_lr': lr}], 
                                                lr=lr, 
                                                betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)
            
            # fp16 training
            if opt.fp16:
                self.scaler = torch.cuda.amp.GradScaler()
            
            # discriminator setting
            if not opt.no_discriminator:
                self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor) 
                self.loss_names_D = ['D_real', 'D_fake']
                # initialize optimizer D
                if opt.TTUR:                
                    beta1, beta2 = 0, 0.9
                    lr = opt.lr * 2
                else:
                    beta1, beta2 = opt.beta1, 0.999
                    lr = opt.lr
                self.optimizer_D = torch.optim.Adam([{'params': self.Feature2Face_D.module.netD.parameters(),
                                                      'initial_lr': lr}], 
                                                    lr=lr, 
                                                    betas=(beta1, beta2))
                self.optimizers.append(self.optimizer_D)
    
    
    def init_paras(self, dataset):
        opt = self.opt
        iter_path = os.path.join(self.save_dir, 'iter.txt')
        start_epoch, epoch_iter = 1, 0
        ### if continue training, recover previous states
        if opt.continue_train:        
            if os.path.exists(iter_path):
                start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
                print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
                # change epoch count & update schedule settings
                opt.epoch_count = start_epoch
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
                # print lerning rate
                lr = self.optimizers[0].param_groups[0]['lr']
                print('update learning rate: {} -> {}'.format(opt.lr, lr))
            else:
                print('not found training log, hence training from epoch 1')
            # change training sequence length
#            if start_epoch > opt.nepochs_step:
#                dataset.dataset.update_training_batch((start_epoch-1)//opt.nepochs_step)  
                

        total_steps = (start_epoch-1) * len(dataset) + epoch_iter
        total_steps = total_steps // opt.print_freq * opt.print_freq  
        
        return start_epoch, opt.print_freq, total_steps, epoch_iter
    
    
    
    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.feature_map, self.cand_image, self.tgt_image, self.facial_mask = \
            data['feature_map'], data['cand_image'], data['tgt_image'], data['weight_mask']
        self.feature_map = self.feature_map.to(self.device)
        self.cand_image = self.cand_image.to(self.device)
        self.tgt_image = self.tgt_image.to(self.device)
#        self.facial_mask = self.facial_mask.to(self.device)


    def forward(self):
        ''' forward pass for feature2Face
        '''  
        self.input_feature_maps = torch.cat([self.feature_map, self.cand_image], dim=1)
        self.fake_pred = self.Feature2Face_G(self.input_feature_maps)
        

        

    def backward_G(self):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.Feature2Face_D(real_AB)
        pred_fake = self.Feature2Face_D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1
#        loss_maskL1 = self.criterionMaskL1(self.fake_pred, self.tgt_image, self.facial_mask * self.opt.lambda_mask)
        loss_vgg, loss_style = self.criterionVGG(self.fake_pred, self.tgt_image, style=True)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat 
        loss_style = torch.mean(loss_style) * self.opt.lambda_feat 
        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)
        
        # combine loss and calculate gradients
        
        if not self.opt.fp16:
            self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM #+ loss_maskL1
            self.loss_G.backward()
        else:
            with autocast():
                self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM #+ loss_maskL1
            self.scaler.scale(self.loss_G).backward()
        
        self.loss_dict = {**self.loss_dict, **dict(zip(self.loss_names_G, [loss_l1, loss_vgg, loss_style, loss_G_GAN, loss_FM]))}

        
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.Feature2Face_D(real_AB)
        pred_fake = self.Feature2Face_D(fake_AB.detach())
        with autocast():
            loss_D_real = self.criterionGAN(pred_real, True) * 2
            loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5 
        
        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))   
        
        if not self.opt.fp16:
            self.loss_D.backward()
        else:
            self.scaler.scale(self.loss_D).backward()
    
    
    def compute_FeatureMatching_loss(self, pred_fake, pred_real):
        # GAN feature matching loss
        loss_FM = torch.zeros(1).cuda()
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(min(len(pred_fake), self.opt.num_D)):
            for j in range(len(pred_fake[i])):
                loss_FM += D_weights * feat_weights * \
                    self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        
        return loss_FM
    
    
    
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        # only train single image generation
        ## forward
        self.forward()
        # update D
        self.set_requires_grad(self.Feature2Face_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero    
        if not self.opt.fp16:
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        else:
            with autocast():
                self.backward_D()
            self.scaler.step(self.optimizer_D)
            
            
        # update G
        self.set_requires_grad(self.Feature2Face_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if not self.opt.fp16:
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights   
        else:
            with autocast():
                self.backward_G()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
            

    def inference(self, feature_map, cand_image):
        """ inference process """
        with torch.no_grad():      
            if cand_image == None:
                input_feature_maps = feature_map
            else:
                input_feature_maps = torch.cat([feature_map, cand_image], dim=1)
            if not self.opt.fp16:
                fake_pred = self.Feature2Face_G(input_feature_maps)          
            else:
                with autocast():
                    fake_pred = self.Feature2Face_G(input_feature_maps) 
        return fake_pred










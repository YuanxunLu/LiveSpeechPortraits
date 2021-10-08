import numpy as np
import torch

from .base_model import BaseModel
from . import networks
from . import audio2feature





class Audio2FeatureModel(BaseModel):          
    def __init__(self, opt):
        """Initialize the Audio2Feature class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['Audio2Feature']
        self.Audio2Feature = networks.init_net(audio2feature.Audio2Feature(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
                
        # define only during training time
        if self.isTrain:
            # losses
            self.featureL2loss = torch.nn.MSELoss().to(self.device)
            # optimizer
            self.optimizer = torch.optim.Adam([{'params':self.Audio2Feature.parameters(), 
                                                'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.99))

            self.optimizers.append(self.optimizer)
            
            if opt.continue_train:
                self.resume_training()

    
    def resume_training(self):
        opt = self.opt
        ### if continue training, recover previous states            
        print('Resuming from epoch %s ' % (opt.load_epoch))   
        # change epoch count & update schedule settings
        opt.epoch_count = int(opt.load_epoch)
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # print lerning rate
        lr = self.optimizers[0].param_groups[0]['lr']
        print('update learning rate: {} -> {}'.format(opt.lr, lr))
    
    
    
    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """

        self.audio_feats, self.target_info = data
#        b, item_length, mel_channels, width = self.audio_feats.shape
        self.audio_feats = self.audio_feats.to(self.device)  
        self.target_info = self.target_info.to(self.device)        
        
        # gaussian noise
#        if self.opt.gaussian_noise:
#            self.audio_feats = self.opt.gaussian_noise_scale * torch.randn(self.audio_feats.shape).cuda()
#            self.target_info += self.opt.gaussian_noise_scale * torch.randn(self.target_info.shape).cuda()
   
                
    
    def forward(self):
        '''
        Args:
            history_landmarks: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        Returns:
            preds: [b, T, output_channels]
        '''  
        self.preds = self.Audio2Feature.forward(self.audio_feats)  

        
    
    def calculate_loss(self):
        """ calculate loss in detail, only forward pass included""" 
        if self.opt.loss == 'GMM':
            b, T, _ = self.target_info.shape
            self.loss_GMM = self.criterion_GMM(self.preds, self.target_info)
            self.loss = self.loss_GMM

        elif self.opt.loss == 'L2':
            frame_future = self.opt.frame_future
            if not frame_future == 0:
                self.loss = self.featureL2loss(self.preds[:, frame_future:], self.target_info[:, :-frame_future]) * 1000
            else:
                self.loss = self.featureL2loss(self.preds, self.target_info) * 1000

    
    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_loss()
        self.loss.backward()
        
    
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.optimizer.zero_grad()   # clear optimizer parameters grad
        self.forward()               # forward pass
        self.backward()              # calculate loss and gradients
        self.optimizer.step()        # update gradients 
            
    
    def validate(self):
        """ validate process """
        with torch.no_grad():
            self.forward()
            self.calculate_loss()
    
    
    def generate_sequences(self, audio_feats, sample_rate = 16000, fps=60, fill_zero=True, opt=[]):
        ''' generate landmark sequences given audio and a initialized landmark.
        Note that the audio input should have the same sample rate as the training.
        Args:
            audio_sequences: [n,], in numpy
            init_landmarks: [npts, 2], in numpy
            sample_rate: audio sample rate, should be same as training process.
            method(str): optional, how to generate the sequence, indeed it is the 
                loss function during training process. Options are 'L2' or 'GMM'.
        Reutrns:
            landmark_sequences: [T, npts, 2] predition landmark sequences
        '''

        frame_future = opt.frame_future
        nframe = int(audio_feats.shape[0] / 2) 
        
        if not frame_future == 0:
            audio_feats_insert = np.repeat(audio_feats[-1],  2 * (frame_future)).reshape(-1, 2 * (frame_future)).T
            audio_feats = np.concatenate([audio_feats, audio_feats_insert])
        
                       
        # evaluate mode
        self.Audio2Feature.eval()
                
        with torch.no_grad():
            input = torch.from_numpy(audio_feats).unsqueeze(0).float().to(self.device)
            preds = self.Audio2Feature.forward(input)
            
            # drop first frame future results
        if not frame_future == 0:
            preds = preds[0, frame_future:].cpu().detach().numpy()
        else:
            preds = preds[0, :].cpu().detach().numpy()
        
        assert preds.shape[0] == nframe
                            

        return preds
    
    
                                
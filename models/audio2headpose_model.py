import numpy as np
import torch
from tqdm import tqdm

from .base_model import BaseModel
from . import networks
from . import audio2headpose
from .losses import GMMLogLoss, Sample_GMM
import torch.nn as nn



class Audio2HeadposeModel(BaseModel):          
    def __init__(self, opt):
        """Initialize the Audio2Headpose class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['Audio2Headpose']
        if opt.feature_decoder == 'WaveNet':
            self.Audio2Headpose = networks.init_net(audio2headpose.Audio2Headpose(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        elif opt.feature_decoder == 'LSTM':
            self.Audio2Headpose = networks.init_net(audio2headpose.Audio2Headpose_LSTM(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
             
        # define only during training time
        if self.isTrain:
            # losses
            self.criterion_GMM = GMMLogLoss(opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, opt.A2H_GMM_sigma_min).to(self.device)
            self.criterion_L2 = nn.MSELoss().cuda()
            # optimizer
            self.optimizer = torch.optim.Adam([{'params':self.Audio2Headpose.parameters(), 
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
        if self.opt.feature_decoder == 'WaveNet':
            self.headpose_audio_feats, self.history_headpose, self.target_headpose = data      
            self.headpose_audio_feats = self.headpose_audio_feats.to(self.device)
            self.history_headpose = self.history_headpose.to(self.device)
            self.target_headpose = self.target_headpose.to(self.device)     
        elif self.opt.feature_decoder == 'LSTM':
            self.headpose_audio_feats, self.target_headpose = data      
            self.headpose_audio_feats = self.headpose_audio_feats.to(self.device)
            self.target_headpose = self.target_headpose.to(self.device)    
        
                
    
    def forward(self):
        '''
        Args:
            history_landmarks: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        Returns:
            preds: [b, T, output_channels]
        '''  
        
        if self.opt.audio_windows == 2:
            bs, item_len, ndim = self.headpose_audio_feats.shape
            self.headpose_audio_feats = self.headpose_audio_feats.reshape(bs, -1, ndim * 2)
        else:
            bs, item_len, _, ndim = self.headpose_audio_feats.shape
        if self.opt.feature_decoder == 'WaveNet':
            self.preds_headpose = self.Audio2Headpose.forward(self.history_headpose, self.headpose_audio_feats) 
        elif self.opt.feature_decoder == 'LSTM':
            self.preds_headpose = self.Audio2Headpose.forward(self.headpose_audio_feats) 
    
    
    def calculate_loss(self):
        """ calculate loss in detail, only forward pass included""" 
        if self.opt.loss == 'GMM':
            self.loss_GMM = self.criterion_GMM(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_GMM
        elif self.opt.loss == 'L2':
            self.loss_L2 = self.criterion_L2(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_L2
        
        if not self.opt.smooth_loss == 0:                    
            mu_gen = Sample_GMM(self.preds_headpose, 
                                self.Audio2Headpose.module.WaveNet.ncenter, 
                                self.Audio2Headpose.module.WaveNet.ndim, 
                                sigma_scale=0)
            self.smooth_loss = (mu_gen[:,2:] + self.target_headpose[:,:-2] - 2 * self.target_headpose[:,1:-1]).mean(dim=2).abs().mean()
            self.loss += self.smooth_loss * self.opt.smooth_loss

            
    
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
    
    
    def generate_sequences(self, audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.0, opt=[]):
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
        audio_feats = audio_feats.reshape(-1, 512 * 2)
        nframe = audio_feats.shape[0] - frame_future        
        pred_headpose = np.zeros([nframe, opt.A2H_GMM_ndim])
        
        if opt.feature_decoder == 'WaveNet':
            # fill zero or not
            if fill_zero == True:
                # headpose
                audio_feats_insert = np.repeat(audio_feats[0], opt.A2H_receptive_field - 1)
                audio_feats_insert = audio_feats_insert.reshape(-1, opt.A2H_receptive_field - 1).T
                audio_feats = np.concatenate([audio_feats_insert, audio_feats])
                # history headpose
                history_headpose = np.repeat(pre_headpose, opt.A2H_receptive_field)
                history_headpose = history_headpose.reshape(-1, opt.A2H_receptive_field).T
                history_headpose = torch.from_numpy(history_headpose).unsqueeze(0).float().to(self.device)
                infer_start = 0   
            else:
                return None     
                      
            # evaluate mode
            self.Audio2Headpose.eval()
                    
            with torch.no_grad():
                for i in tqdm(range(infer_start, nframe), desc='generating headpose'):
                    history_start = i - infer_start
                    input_audio_feats = audio_feats[history_start + frame_future: history_start + frame_future + opt.A2H_receptive_field]
                    input_audio_feats = torch.from_numpy(input_audio_feats).unsqueeze(0).float().to(self.device) 

                    if self.opt.feature_decoder == 'WaveNet':
                        preds = self.Audio2Headpose.forward(history_headpose, input_audio_feats) 
                    elif self.opt.feature_decoder == 'LSTM':
                        preds = self.Audio2Headpose.forward(input_audio_feats) 
                         
                    if opt.loss == 'GMM':
                        pred_data = Sample_GMM(preds, opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, sigma_scale=sigma_scale)  
                    elif opt.loss == 'L2':
                        pred_data = preds
                        
                    # get predictions
                    pred_headpose[i] = pred_data[0,0].cpu().detach().numpy()  
                    history_headpose = torch.cat((history_headpose[:,1:,:], pred_data.to(self.device)), dim=1)  # add in time-axis                
    
            return pred_headpose
        
        elif opt.feature_decoder == 'LSTM':
            self.Audio2Headpose.eval()
            with torch.no_grad():
                input = torch.from_numpy(audio_feats).unsqueeze(0).float().to(self.device)
                preds = self.Audio2Headpose.forward(input)
                if opt.loss == 'GMM':
                    pred_data = Sample_GMM(preds, opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, sigma_scale=sigma_scale) 
                elif opt.loss == 'L2':
                    pred_data = preds
                # get predictions
                pred_headpose = pred_data[0].cpu().detach().numpy()  
            
            return pred_headpose
    
    
            
    
    
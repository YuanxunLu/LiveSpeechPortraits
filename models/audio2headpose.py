import torch.nn as nn

from .networks import WaveNet



class Audio2Headpose(nn.Module):
    def __init__(self, opt):
        super(Audio2Headpose, self).__init__()
        self.opt = opt
        if self.opt.loss == 'GMM':
            output_size = (2 * opt.A2H_GMM_ndim + 1) * opt.A2H_GMM_ncenter
        elif self.opt.loss == 'L2':
            output_size = opt.A2H_GMM_ndim
        # define networks   
        self.audio_downsample = nn.Sequential(
                        nn.Linear(in_features=opt.APC_hidden_size * 2, out_features=opt.APC_hidden_size),
                        nn.BatchNorm1d(opt.APC_hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(opt.APC_hidden_size, opt.APC_hidden_size),
                        )
        
        self.WaveNet = WaveNet(opt.A2H_wavenet_residual_layers,
                               opt.A2H_wavenet_residual_blocks,
                               opt.A2H_wavenet_residual_channels,
                               opt.A2H_wavenet_dilation_channels,
                               opt.A2H_wavenet_skip_channels,
                               opt.A2H_wavenet_kernel_size,
                               opt.time_frame_length,
                               opt.A2H_wavenet_use_bias,
                               True,
                               opt.A2H_wavenet_input_channels,
                               opt.A2H_GMM_ncenter,
                               opt.A2H_GMM_ndim,
                               output_size,
                               opt.A2H_wavenet_cond_channels)
        self.item_length = self.WaveNet.receptive_field + opt.time_frame_length - 1
                    

    def forward(self, history_info, audio_features):
        '''
        Args:
            history_info: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        '''
        # APC features: [b, item_length, APC_hidden_size] ==> [b, APC_hidden_size, item_length]
        bs, item_len, ndim = audio_features.shape
        down_audio_feats = self.audio_downsample(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)
        pred = self.WaveNet.forward(history_info.permute(0,2,1), down_audio_feats.transpose(1,2)) 


        return pred
    



class Audio2Headpose_LSTM(nn.Module):
    def __init__(self, opt):
        super(Audio2Headpose_LSTM, self).__init__()
        self.opt = opt
        if self.opt.loss == 'GMM':
            output_size = (2 * opt.A2H_GMM_ndim + 1) * opt.A2H_GMM_ncenter
        elif self.opt.loss == 'L2':
            output_size = opt.A2H_GMM_ndim
        # define networks         
        self.audio_downsample = nn.Sequential(
                        nn.Linear(in_features=opt.APC_hidden_size * 2, out_features=opt.APC_hidden_size),
                        nn.BatchNorm1d(opt.APC_hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(opt.APC_hidden_size, opt.APC_hidden_size),
                        )
        
        self.LSTM = nn.LSTM(input_size=opt.APC_hidden_size,
                            hidden_size=256,
                            num_layers=3,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(in_features=256, out_features=512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, output_size))
                    

    def forward(self, audio_features):
        '''
        Args:
            history_info: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        '''
        # APC features: [b, item_length, APC_hidden_size] ==> [b, APC_hidden_size, item_length]
        bs, item_len, ndim = audio_features.shape
        down_audio_feats = self.audio_downsample(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)
        output, (hn, cn) = self.LSTM(down_audio_feats)
        pred = self.fc(output.reshape(-1, 256)).reshape(bs, item_len, -1)


        return pred





    
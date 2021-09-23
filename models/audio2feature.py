import torch.nn as nn
from .networks import WaveNet



class Audio2Feature(nn.Module):
    def __init__(self, opt):
        super(Audio2Feature, self).__init__()
        self.opt = opt
        opt.A2L_wavenet_input_channels = opt.APC_hidden_size
        if self.opt.loss == 'GMM':
            output_size = (2 * opt.A2L_GMM_ndim + 1) * opt.A2L_GMM_ncenter
        elif self.opt.loss == 'L2':
            num_pred = opt.predict_length
            output_size = opt.A2L_GMM_ndim * num_pred
        # define networks
        if opt.feature_decoder == 'WaveNet':
            self.WaveNet = WaveNet(opt.A2L_wavenet_residual_layers,
                                   opt.A2L_wavenet_residual_blocks,
                                   opt.A2L_wavenet_residual_channels,
                                   opt.A2L_wavenet_dilation_channels,
                                   opt.A2L_wavenet_skip_channels,
                                   opt.A2L_wavenet_kernel_size,
                                   opt.time_frame_length,
                                   opt.A2L_wavenet_use_bias,
                                   opt.A2L_wavenet_cond,
                                   opt.A2L_wavenet_input_channels,
                                   opt.A2L_GMM_ncenter,
                                   opt.A2L_GMM_ndim,
                                   output_size)
            self.item_length = self.WaveNet.receptive_field + opt.time_frame_length - 1
        elif opt.feature_decoder == 'LSTM':
            self.downsample = nn.Sequential(
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
            audio_features: [b, T, ndim]
        '''
        if self.opt.feature_decoder == 'WaveNet':
            pred = self.WaveNet.forward(audio_features.permute(0,2,1)) 
        elif self.opt.feature_decoder == 'LSTM':
            bs, item_len, ndim = audio_features.shape
            # new in 0324
            audio_features = audio_features.reshape(bs, -1, ndim*2)
            down_audio_feats = self.downsample(audio_features.reshape(-1, ndim*2)).reshape(bs, int(item_len/2), ndim)
            output, (hn, cn) = self.LSTM(down_audio_feats)
#            output, (hn, cn) = self.LSTM(audio_features)
            pred = self.fc(output.reshape(-1, 256)).reshape(bs, int(item_len/2), -1)
#            pred = self.fc(output.reshape(-1, 256)).reshape(bs, item_len, -1)[:, -self.opt.time_frame_length:, :] 
        
        return pred













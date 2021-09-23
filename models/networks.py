import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import init
import functools


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



###############################################################################
# The detailed network architecture implementation for each model
###############################################################################

class APC_encoder(nn.Module):
    def __init__(self,
                 mel_dim,
                 hidden_size,
                 num_layers,
                 residual):
        super(APC_encoder, self).__init__()

        input_size = mel_dim

        in_sizes = ([input_size] + [hidden_size] * (num_layers - 1))
        out_sizes = [hidden_size] * num_layers
        self.rnns = nn.ModuleList(
                [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_residual = residual
    
    def forward(self, inputs, lengths):
        '''
        input:
            inputs: (batch_size, seq_len, mel_dim)
            lengths: (batch_size,)

        return:
            predicted_mel: (batch_size, seq_len, mel_dim)
            internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
            where x is 1 if there's a prenet, otherwise 0
        '''
        with torch.no_grad():
            seq_len = inputs.size(1)
            packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)
        
            for i, layer in enumerate(self.rnns):
                packed_rnn_outputs, _ = layer(packed_rnn_inputs)
                
                rnn_outputs, _ = pad_packed_sequence(
                        packed_rnn_outputs, True, total_length=seq_len)
                # outputs: (batch_size, seq_len, rnn_hidden_size)
                
                if i + 1 < len(self.rnns):
                    rnn_inputs, _ = pad_packed_sequence(
                            packed_rnn_inputs, True, total_length=seq_len)
                    # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)
                    if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                        # Residual connections
                        rnn_outputs = rnn_outputs + rnn_inputs
                    packed_rnn_inputs = pack_padded_sequence(rnn_outputs, lengths, True)
        
        
        return rnn_outputs




class WaveNet(nn.Module):
    ''' This is a complete implementation of WaveNet architecture, mainly composed
    of several residual blocks and some other operations.
    Args:
        batch_size: number of batch size
        residual_layers: number of layers in each residual blocks
        residual_blocks: number of residual blocks
        dilation_channels: number of channels for the dilated convolution
        residual_channels: number of channels for the residual connections
        skip_channels: number of channels for the skip connections
        end_channels: number of channels for the end convolution
        classes: Number of possible values each sample can have as output
        kernel_size: size of dilation convolution kernel
        output_length(int): Number of samples that are generated for each input
        use_bias: whether bias is used in each layer.
        cond(bool): whether condition information are applied. if cond == True:
            cond_channels: channel number of condition information
        `` loss(str): GMM loss is adopted. ``
    '''
    def __init__(self,
                 residual_layers = 10,
                 residual_blocks = 3,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 kernel_size = 2,
                 output_length = 16,
                 use_bias = False,
                 cond = True,
                 input_channels = 128,
                 ncenter = 1,
                 ndim = 73*2,
                 output_channels = 73*3,
                 cond_channels = 256,
                 activation = 'leakyrelu'):
        super(WaveNet, self).__init__()
        
        self.layers = residual_layers
        self.blocks = residual_blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.ncenter = ncenter
        self.ndim = ndim
#        self.output_channels = (2 * self.ndim + 1) * self.ncenter
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        
        # build modules
        self.dilations = []
        self.dilation_queues = []
        residual_blocks = []
        self.receptive_field = 1
        
        # 1x1 convolution to create channels
        self.start_conv1 = nn.Conv1d(in_channels=self.input_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        self.start_conv2 = nn.Conv1d(in_channels=self.residual_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.drop_out2D = nn.Dropout2d(p=0.5)
        
        
        # build residual blocks
        for b in range(self.blocks):
            new_dilation = 1
            additional_scope = kernel_size - 1
            for i in range(self.layers):
                # create current residual block
                residual_blocks.append(residual_block(dilation = new_dilation,
                                                      dilation_channels = self.dilation_channels,
                                                      residual_channels = self.residual_channels,
                                                      skip_channels = self.skip_channels,
                                                      kernel_size = self.kernel_size,
                                                      use_bias = self.bias,
                                                      cond = self.cond,
                                                      cond_channels = self.cond_channels))
                new_dilation *= 2
                
                self.receptive_field += additional_scope
                additional_scope *= 2
        
        self.residual_blocks = nn.ModuleList(residual_blocks)
        # end convolutions
        
        self.end_conv_1 = nn.Conv1d(in_channels = self.skip_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        self.end_conv_2 = nn.Conv1d(in_channels = self.output_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        
    
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
    
    def forward(self, input, cond=None):
        '''
        Args:
            input: [b, ndim, T]
            cond: [b, nfeature, T]
        Returns:
            res: [b, T, ndim]
        '''
        # dropout
        x = self.drop_out2D(input)
        
        # preprocess
        x = self.activation(self.start_conv1(x))
        x = self.activation(self.start_conv2(x))
        skip = 0
#        for i in range(self.blocks * self.layers):
        for i, dilation_block in enumerate(self.residual_blocks):
            x, current_skip = self.residual_blocks[i](x, cond)
            skip += current_skip
        
        # postprocess
        res = self.end_conv_1(self.activation(skip))
        res = self.end_conv_2(self.activation(res))
        
        # cut the output size
        res = res[:, :, -self.output_length:]  # [b, ndim, T]
        res = res.transpose(1, 2)  # [b, T, ndim]
        
        return res
    
    
    
class residual_block(nn.Module):
    '''
    This is the implementation of a residual block in wavenet model. Every
    residual block takes previous block's output as input. The forward pass of 
    each residual block can be illusatrated as below:
        
    ######################### Current Residual Block ##########################
    #     |-----------------------*residual*--------------------|             #
    #     |                                                     |             # 
    #     |        |-- dilated conv -- tanh --|                 |             #
    # -> -|-- pad--|                          * ---- |-- 1x1 -- + --> *input* #
    #              |-- dilated conv -- sigm --|      |                        #
    #                                               1x1                       # 
    #                                                |                        # 
    # ---------------------------------------------> + -------------> *skip*  #
    ###########################################################################
    As shown above, each residual block returns two value: 'input' and 'skip':
        'input' is indeed this block's output and also is the next block's input.
        'skip' is the skip data which will be added finally to compute the prediction.
    The input args own the same meaning in the WaveNet class.
    
    '''
    def __init__(self,
                 dilation,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 kernel_size = 2,
                 use_bias = False,
                 cond = True,
                 cond_channels = 128):
        super(residual_block, self).__init__()
        
        self.dilation = dilation
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        # zero padding to the left of the sequence.
        self.padding = (int((self.kernel_size - 1) * self.dilation), 0)
        
        # dilated convolutions
        self.filter_conv= nn.Conv1d(in_channels = self.residual_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = self.kernel_size,
                                    dilation = self.dilation,
                                    bias = self.bias)
                
        self.gate_conv = nn.Conv1d(in_channels = self.residual_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = self.kernel_size,
                                   dilation = self.dilation,
                                   bias = self.bias)
                
        # 1x1 convolution for residual connections
        self.residual_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                       out_channels = self.residual_channels,
                                       kernel_size = 1,
                                       bias = self.bias)
                
        # 1x1 convolution for skip connections
        self.skip_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                   out_channels = self.skip_channels,
                                   kernel_size = 1,
                                   bias = self.bias)
        
        # condition conv, no dilation
        if self.cond == True:
            self.cond_filter_conv = nn.Conv1d(in_channels = self.cond_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = 1,
                                    bias = True)
            self.cond_gate_conv = nn.Conv1d(in_channels = self.cond_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = 1,
                                   bias = True)
        
    
    def forward(self, input, cond=None):
        if self.cond is True and cond is None:
            raise RuntimeError("set using condition to true, but no cond tensor inputed")
            
        x_pad = F.pad(input, self.padding)
        # filter
        filter = self.filter_conv(x_pad)
        # gate
        gate = self.gate_conv(x_pad)
        
        if self.cond == True and cond is not None:
            filter_cond = self.cond_filter_conv(cond)
            gate_cond = self.cond_gate_conv(cond)
            # add cond results
            filter = filter + filter_cond
            gate = gate + gate_cond
                       
        # element-wise multiple
        filter = torch.tanh(filter)
        gate = torch.sigmoid(gate)
        x = filter * gate
        
        # residual and skip
        residual = self.residual_conv(x) + input
        skip = self.skip_conv(x)
               
        
        return residual, skip




## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
            )
    


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], useDDP=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if useDDP:
            net = net().to(gpu_ids)
            net = DDP(net, device_ids=gpu_ids)  # DDP
            print(f'use DDP to apply models on {gpu_ids}')
        else:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=opt.epoch_count-2)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.gamma, last_epoch=opt.epoch_count-2)
        for _ in range(opt.epoch_count-2):
            scheduler.step()
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    


        
class Feature2FaceGenerator_normal(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Feature2FaceGenerator_normal, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock_small(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock_small(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        output = torch.tanh(output)   # scale to [-1, 1]

        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock_small(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock_small, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer)]

        # res_downconv = [ResidualBlock(inner_nc)]
        # res_upconv = [ResidualBlock(outer_nc)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

   

class Feature2FaceGenerator_large(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Feature2FaceGenerator_large, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        output = torch.tanh(output)   # scale to [-1, 1]

        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        # res_downconv = [ResidualBlock(inner_nc)]
        # res_upconv = [ResidualBlock(outer_nc)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            # up = [uprelu, upsample, upconv, upnorm]
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# UNet with residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            # hard to converge with out batch or instance norm
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out
        # return self.relu(x + self.block(x))



class Feature2FaceGenerator_Unet(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Feature2FaceGenerator_Unet, self).__init__()
        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer


    def forward(self, input):
        output = self.model(input)

        return output




class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf), 
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            


    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)      


  
    
    


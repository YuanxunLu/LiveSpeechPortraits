import argparse
import os
from util import util
import torch
import models
import numpy as np


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ## task
        parser.add_argument('--task', type=str, default='Audio2Headpose', help='|Audio2Feature|Feature2Face|Full|')
        

        ## basic parameters
        parser.add_argument('--model', type=str, default='audio2headpose', help='trained model')
        parser.add_argument('--dataset_mode', type=str, default='audiovisual', help='chooses how datasets are loaded. [unaligned | aligned | single]') 
        parser.add_argument('--name', type=str, default='Audio2Headpose', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        
        
        # data parameters
        parser.add_argument('--FPS', type=str, default=60, help='video fps')
        parser.add_argument('--sample_rate', type=int, default=16000, help='audio sample rate')
        parser.add_argument('--audioRF_history', type=int, default=60, help='audio history receptive field length')
        parser.add_argument('--audioRF_future', type=int, default=0, help='audio future receptive field length')
        parser.add_argument('--feature_decoder', type=str, default='WaveNet', help='|WaveNet|LSTM|')
        parser.add_argument('--loss', type=str, default='GMM', help='|GMM|L2|') 

        
        # dataset parameters
        parser.add_argument('--dataset_names', type=str, default='name', help='chooses how datasets are loaded.')
        parser.add_argument('--dataroot', type=str, default='path')
        parser.add_argument('--frame_jump_stride', type=int, default=1, help='jump index in audio dataset.')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--audio_encoder', type=str, default='APC', help='|CNN|LSTM|APC|')
        parser.add_argument('--audiofeature_input_channels', type=int, default=80, help='input channels of audio features')
        parser.add_argument('--frame_future', type=int, default=15)
        parser.add_argument('--predict_length', type=int, default=5)
        parser.add_argument('--audio_windows', type=int, default=2)
        parser.add_argument('--time_frame_length', type=int, default=240, help='length of training frames in each iteration')
        
        
        # APC parameters
        parser.add_argument('--APC_hidden_size', type=int, default=512)
        parser.add_argument('--APC_rnn_layers', type=int, default=3)
        parser.add_argument("--APC_residual", action="store_true")
        parser.add_argument('--APC_frame_history', type=int, default=60)
            
        
        ## network parameters    
        # audio2headpose wavenet 
        parser.add_argument('--A2H_wavenet_residual_layers', type=int, default=7, help='residual layer numbers')
        parser.add_argument('--A2H_wavenet_residual_blocks', type=int, default=2, help='residual block numbers')
        parser.add_argument('--A2H_wavenet_dilation_channels', type=int, default=128, help='dilation convolution channels')
        parser.add_argument('--A2H_wavenet_residual_channels', type=int, default=128, help='residual channels')
        parser.add_argument('--A2H_wavenet_skip_channels', type=int, default=256, help='skip channels')      
        parser.add_argument('--A2H_wavenet_kernel_size', type=int, default=2, help='dilation convolution kernel size')
        parser.add_argument('--A2H_wavenet_use_bias', type=bool, default=True, help='whether to use bias in dilation convolution')
        parser.add_argument('--A2H_wavenet_cond', type=bool, default=True, help='whether use condition input')
        parser.add_argument('--A2H_wavenet_cond_channels', type=int, default=512, help='whether use condition input')
        parser.add_argument('--A2H_wavenet_input_channels', type=int, default=12, help='input channels')
        parser.add_argument('--A2H_GMM_ncenter', type=int, default=1, help='gaussian distribution numbers, 1 for single gaussian distribution')
        parser.add_argument('--A2H_GMM_ndim', type=int, default=12, help='dimension of each gaussian, usually number of pts')
        parser.add_argument('--A2H_GMM_sigma_min', type=float, default=0.03, help='minimal gaussian sigma values')
        
        
        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--sequence_length', type=int, default=240, help='length of training frames in each iteration')
        
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults


        # save and return the parser
        self.parser = parser
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if opt.isTrain:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])
        
        # set datasets
        if self.isTrain:
            opt.train_dataset_names = np.loadtxt(os.path.join(opt.dataroot, 
                                                              opt.dataset_names, 
                                                              opt.train_dataset_names), dtype=np.str).tolist()
            if type(opt.train_dataset_names) == str:
                opt.train_dataset_names = [opt.train_dataset_names]
            opt.validate_dataset_names = np.loadtxt(os.path.join(opt.dataroot, 
                                                                 opt.dataset_names, 
                                                                 opt.validate_dataset_names), dtype=np.str).tolist()
            if type(opt.validate_dataset_names) == str:
                opt.validate_dataset_names = [opt.validate_dataset_names]

        self.opt = opt
        return self.opt












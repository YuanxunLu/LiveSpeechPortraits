import argparse
import os
from util import util
import torch
import numpy as np

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):  
        ## task
        self.parser.add_argument('--task', type=str, default='Feature2Face', help='|Audio2Feature|Feature2Face|Full|')  
        self.parser.add_argument('--model', type=str, default='feature2face', help='chooses which model to use. vid2vid, test')                 
        self.parser.add_argument('--name', type=str, default='TestRender', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        
        # display
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')        
        self.parser.add_argument('--tf_log', default=True, action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        
        
        # input/output size
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')      
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')  
              
                                 
        # setting inputs
        self.parser.add_argument('--dataset_mode', type=str, default='face', help='chooses how datasets are loaded.')
        self.parser.add_argument('--dataroot', type=str, default='./data/') 
        self.parser.add_argument('--isH5', type=int, default=1, help='whether to use h5py to save dataset')
        self.parser.add_argument('--suffix', type=str, default='.jpg', help='image suffix')
        self.parser.add_argument('--isMask', type=int, default=0, help='use face mask')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='scaleWidth', help='scaling and cropping of images at load time [resize_and_crop|crop|scaledCrop|scaleWidth|scaleWidth_and_crop|scaleWidth_and_scaledCrop|scaleHeight|scaleHeight_and_crop] etc')
        self.parser.add_argument('--no_flip', type=int, default=1, help='if specified, do not flip the images for data argumentation')    


        # generator arch  
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')       
        self.parser.add_argument('--n_downsample_G', type=int, default=8, help='number of downsampling layers in netG')        
        self.parser.add_argument('--ngf_E', type=int, default=16, help='# of gen filters in first conv layer')       
        self.parser.add_argument('--n_downsample_E', type=int, default=3, help='number of downsampling layers in Enhancement')
        self.parser.add_argument('--n_blocks_E', type=int, default=3, help='number of resnet blocks in Enhancement')
        
        
        # miscellaneous                
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')                
        self.parser.add_argument('--debug', action='store_true', help='if specified, use small dataset for debug')
        self.parser.add_argument('--fp16', type=int, default=0, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        self.initialized = True

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.isTrain = self.isTrain   # train or test
        
        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)
        
        # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])
        
        # set datasets
        datasets = self.opt.dataset_names.split(',')
        self.opt.dataset_names = []
        for name in datasets:
            self.opt.dataset_names.append(name)
        
        if self.isTrain:
            self.opt.train_dataset_names = np.loadtxt(os.path.join(self.opt.dataroot, 
                                                                   self.opt.dataset_names[0], 
                                                                   self.opt.train_dataset_names), dtype=np.str).tolist()
            if type(self.opt.train_dataset_names) == str:
                self.opt.train_dataset_names = [self.opt.train_dataset_names]
            self.opt.validate_dataset_names = np.loadtxt(os.path.join(self.opt.dataroot, 
                                                                      self.opt.dataset_names[0], 
                                                                      self.opt.validate_dataset_names), dtype=np.str).tolist()
            if type(self.opt.validate_dataset_names) == str:
                self.opt.validate_dataset_names = [self.opt.validate_dataset_names]
            
        else:
            test_datasets = self.opt.test_dataset_names.split(',')
            self.opt.test_dataset_names = []
            for name in test_datasets:
                self.opt.test_dataset_names.append(name)
        
        
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        if self.isTrain:
            # save to the disk        
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt

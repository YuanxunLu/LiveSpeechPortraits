from .base_options_feature2face import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        ## dataset settings
        self.parser.add_argument('--dataset_names', type=str, default='name', help='chooses how datasets are loaded.')
        self.parser.add_argument('--train_dataset_names', type=str, default='train_list.txt')
        self.parser.add_argument('--validate_dataset_names', type=str, default='val_list.txt')
        
        
        ## training flags
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen(iterations)')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console(epochs)')
        self.parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of to save the latest results(iterations)')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', default=True, action='store_true', help='continue training: load the latest model')        
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--n_epochs_warm_up', type=int, default=5, help='number of epochs warm up')
        self.parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
        self.parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int, default=900, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.25, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--TTUR', action='store_true', help='Use TTUR training scheme')        
        self.parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        self.parser.add_argument('--pool_size', type=int, default=1, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--frame_jump', type=int, default=1, help='jump frame for training, 1 for not jump')
        self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--seq_max_len', type=int, default=120, help='maximum sequence clip frames sent to network per iteration')
          
        
        # for discriminators 
        self.parser.add_argument('--no_discriminator', type=int, default=0, help='not use discriminator')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--num_D', type=int, default=2, help='number of patch scales in each discriminator')        
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
        self.parser.add_argument('--no_vgg', action='store_true', help='do not use VGG feature matching loss')        
        self.parser.add_argument('--no_ganFeat', action='store_true', help='do not match discriminator features')        
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching')        
        self.parser.add_argument('--sparse_D', action='store_true', help='use sparse temporal discriminators to save memory')


        # for temporal
        self.parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for temporal loss')
        self.parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for temporal loss')
        self.parser.add_argument('--lambda_F', type=float, default=10.0, help='weight for flow loss')
        self.parser.add_argument('--lambda_mask', type=float, default=500.0, help='weight for mask l1 loss')
        self.parser.add_argument('--n_frames_D', type=int, default=3, help='number of frames to feed into temporal discriminator')        
        self.parser.add_argument('--n_scales_temporal', type=int, default=2, help='number of temporal scales in the temporal discriminator')        
        self.parser.add_argument('--n_frames_per_gpu', type=int, default=1, help='the number of frames to load into one GPU at a time. only 1 is supported now')
        self.parser.add_argument('--max_frames_backpropagate', type=int, default=1, help='max number of frames to backpropagate') 
        self.parser.add_argument('--max_t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        self.parser.add_argument('--n_frames_total', type=int, default=12, help='the overall number of frames in a sequence to train with')                
        self.parser.add_argument('--nepochs_step', type=int, default=5, help='how many epochs do we change training sequence length again')
        self.parser.add_argument('--nepochs_fix_global', type=int, default=0, help='if specified, only train the finest spatial layer for the given iterations')

        self.isTrain = True

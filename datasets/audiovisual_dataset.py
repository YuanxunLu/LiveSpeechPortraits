import sys
sys.path.append("..")

from datasets.base_dataset import BaseDataset
import scipy.io as sio
import torch
import librosa
import bisect
import os
import numpy as np
from models.networks import  APC_encoder

from funcs import utils



class AudioVisualDataset(BaseDataset):
    """ audio-visual dataset. currently, return 2D info and 3D tracking info.
    
        # for wavenet:
        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | | 
        
    """
    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.isTrain = self.opt.isTrain
        self.state = opt.dataset_type
        self.dataset_name = opt.dataset_names
        self.target_length = opt.time_frame_length
        self.sample_rate = opt.sample_rate
        self.fps = opt.FPS
        
        self.audioRF_history = opt.audioRF_history
        self.audioRF_future = opt.audioRF_future
        self.compute_mel_online = opt.compute_mel_online
        self.feature_name = opt.feature_name
    
        self.audio_samples_one_frame = self.sample_rate / self.fps
        self.frame_jump_stride = opt.frame_jump_stride
        self.augment = False
        self.task = opt.task
        self.item_length_audio = int((self.audioRF_history + self.audioRF_future)/ self.fps * self.sample_rate)
        
        if self.task == 'Audio2Feature':
            if opt.feature_decoder == 'WaveNet':
                self.A2L_receptive_field = opt.A2L_receptive_field
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
            elif opt.feature_decoder == 'LSTM':
                self.A2L_receptive_field = 30
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
        elif self.task == 'Audio2Headpose':
            self.A2H_receptive_field = opt.A2H_receptive_field
            self.A2H_item_length = self.A2H_receptive_field + self.target_length - 1
            self.audio_window = opt.audio_windows
            self.half_audio_win = int(self.audio_window / 2)
        
        self.frame_future = opt.frame_future
        self.predict_length = opt.predict_length
        self.predict_len = int((self.predict_length - 1) / 2)

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        print('self.device:', self.device)
        if self.task == 'Audio2Feature':
            self.seq_len = opt.sequence_length

        self.total_len = 0     
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        if self.state == 'Train':
            self.clip_names = opt.train_dataset_names
        elif self.state == 'Val':
            self.clip_names = opt.validate_dataset_names
        elif self.state == 'Test':
            self.clip_names = opt.test_dataset_names
            
            
        self.clip_nums = len(self.clip_names)
        # main info
        self.audio = [''] * self.clip_nums
        self.audio_features = [''] * self.clip_nums
        self.feats = [''] * self.clip_nums
        self.exps = [''] * self.clip_nums
        self.pts3d = [''] * self.clip_nums
        self.rot_angles = [''] * self.clip_nums
        self.trans = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums
        self.velocity_pose = [''] * self.clip_nums
        self.acceleration_pose = [''] * self.clip_nums
        self.mean_trans = [''] * self.clip_nums
        if self.state == 'Test':
            self.landmarks = [''] * self.clip_nums
        # meta info
        self.start_point = [''] * self.clip_nums
        self.end_point = [''] * self.clip_nums
        self.len = [''] * self.clip_nums
        self.sample_start = []
        self.clip_valid = ['True'] * self.clip_nums
        self.invalid_clip = []
        
        
        self.mouth_related_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
        if self.task == 'Audio2Feature':
            if self.opt.only_mouth:
                self.indices = self.mouth_related_indices
            else:
                self.indices = np.arange(73)
        if opt.use_delta_pts:
            self.pts3d_mean = np.load(os.path.join(self.dataset_root, 'mean_pts3d.npy'))
        
        for i in range(self.clip_nums):
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)
            # audio
            if os.path.exists(os.path.join(clip_root, name + '_denoise.wav')):
                audio_path = os.path.join(clip_root, name + '_denoise.wav')
                print('find denoised wav!')
            else:
                audio_path = os.path.join(clip_root, name + '.wav')
            self.audio[i], _ = librosa.load(audio_path, sr=self.sample_rate)
                        
            if self.opt.audio_encoder == 'APC':
                APC_name = os.path.split(self.opt.APC_model_path)[-1]
                APC_feature_file = name + '_APC_feature_V0324_ckpt_{}.npy'.format(APC_name)
                APC_feature_path = os.path.join(clip_root, APC_feature_file)
                need_deepfeats = False if os.path.exists(APC_feature_path) else True
                if not need_deepfeats:
                    self.audio_features[i] = np.load(APC_feature_path).astype(np.float32)
            else:
                need_deepfeats = False
                

            # 3D landmarks & headposes
            if self.task == 'Audio2Feature':
                self.start_point[i] = 0
            elif self.task == 'Audio2Headpose':
                self.start_point[i] = 300
            fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
            fit_data = np.load(fit_data_path)
            if not opt.ispts_norm:
                ori_pts3d = fit_data['pts_3d'].astype(np.float32)
            else:
                ori_pts3d = np.load(os.path.join(clip_root, 'tracked3D_normalized_pts_fix_contour.npy'))
            if opt.use_delta_pts:
                self.pts3d[i] = ori_pts3d - self.pts3d_mean
            else:
                self.pts3d[i] = ori_pts3d
            if opt.feature_dtype == 'pts3d':
                self.feats[i] = self.pts3d[i]
            elif opt.feature_dtype == 'FW':
                track_data_path = os.path.join(clip_root, 'tracking_results.mat')
                self.feats[i] = sio.loadmat(track_data_path)['exps'].astype(np.float32)
            self.rot_angles[i] = fit_data['rot_angles'].astype(np.float32)
            # change -180~180 to 0~360
            if not self.dataset_name  == 'Yuxuan':
                rot_change = self.rot_angles[i][:, 0] < 0
                self.rot_angles[i][rot_change, 0] += 360
                self.rot_angles[i][:,0] -= 180   # change x axis direction
            # use delta translation
            self.mean_trans[i] = fit_data['trans'][:,:,0].astype(np.float32).mean(axis=0)
            self.trans[i] = fit_data['trans'][:,:,0].astype(np.float32) - self.mean_trans[i]
            
            self.headposes[i] = np.concatenate([self.rot_angles[i], self.trans[i]], axis=1)
            self.velocity_pose[i] = np.concatenate([np.zeros(6)[None,:], self.headposes[i][1:] - self.headposes[i][:-1]])
            self.acceleration_pose[i] = np.concatenate([np.zeros(6)[None,:], self.velocity_pose[i][1:] - self.velocity_pose[i][:-1]])
            
            if self.dataset_name == 'Yuxuan':
                total_frames = self.feats[i].shape[0] - 300 - 130
            else:
                total_frames = self.feats[i].shape[0] - 60
            
            
            if need_deepfeats:
                if self.opt.audio_encoder == 'APC':
                    print('dataset {} need to pre-compute APC features ...'.format(name))
                    print('first we compute mel spectram for dataset {} '.format(name))                    
                    mel80 = utils.compute_mel_one_sequence(self.audio[i])
                    mel_nframe = mel80.shape[0]
                    print('loading pre-trained model: ', self.opt.APC_model_path)
                    APC_model = APC_encoder(self.opt.audiofeature_input_channels,
                                            self.opt.APC_hidden_size,
                                            self.opt.APC_rnn_layers,
                                            self.opt.APC_residual)
                    APC_model.load_state_dict(torch.load(self.opt.APC_model_path, map_location=str(self.device)), strict=False)
#                    APC_model.load_state_dict(torch.load(self.opt.APC_model_path), strict=False)
                    APC_model.cuda()
                    APC_model.eval()
                    with torch.no_grad():
                        length = torch.Tensor([mel_nframe])
#                        hidden_reps = torch.zeros([mel_nframe, self.opt.APC_hidden_size]).cuda()
                        mel80_torch = torch.from_numpy(mel80.astype(np.float32)).cuda().unsqueeze(0)
                        hidden_reps = APC_model.forward(mel80_torch, length)[0]   # [mel_nframe, 512]
                        hidden_reps = hidden_reps.cpu().numpy()
                        np.save(APC_feature_path, hidden_reps)
                        self.audio_features[i] = hidden_reps


            valid_frames = total_frames - self.start_point[i]
            self.len[i] = valid_frames - 400  
            if i == 0:
                self.sample_start.append(0)
            else:
                self.sample_start.append(self.sample_start[-1] + self.len[i-1] - 1)
            self.total_len += np.int32(np.floor(self.len[i] / self.frame_jump_stride))



    def __getitem__(self, index):
        # recover real index from compressed one
        index_real = np.int32(index * self.frame_jump_stride)
        # find which audio file and the start frame index
        file_index = bisect.bisect_right(self.sample_start, index_real) - 1
        current_frame = index_real - self.sample_start[file_index] + self.start_point[file_index]    
        current_target_length = self.target_length
        
        if self.task == 'Audio2Feature':
            # start point is current frame
            A2Lsamples = self.audio_features[file_index][current_frame * 2 : (current_frame + self.seq_len) * 2]
            target_pts3d = self.feats[file_index][current_frame : current_frame + self.seq_len, self.indices].reshape(self.seq_len, -1)           
            
            A2Lsamples = torch.from_numpy(A2Lsamples).float()
            target_pts3d = torch.from_numpy(target_pts3d).float()
            
            # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
            return A2Lsamples, target_pts3d
            
            
        elif self.task == 'Audio2Headpose':
            if self.opt.feature_decoder == 'WaveNet':
                # find the history info start points
                A2H_history_start = current_frame - self.A2H_receptive_field
                A2H_item_length = self.A2H_item_length
                A2H_receptive_field = self.A2H_receptive_field
                
                if self.half_audio_win == 1:
                    A2Hsamples = self.audio_features[file_index][2 * (A2H_history_start + self.frame_future) : 2 * (A2H_history_start + self.frame_future + A2H_item_length)]
                else:
                    A2Hsamples = np.zeros([A2H_item_length, self.audio_window, 512])
                    for i in range(A2H_item_length):
                        A2Hsamples[i] = self.audio_features[file_index][2 * (A2H_history_start + i) - self.half_audio_win : 2 * (A2H_history_start + i) + self.half_audio_win]
     
                if self.predict_len == 0:
                    target_headpose = self.headposes[file_index][A2H_history_start + A2H_receptive_field : A2H_history_start + A2H_item_length + 1]
                    history_headpose = self.headposes[file_index][A2H_history_start : A2H_history_start + A2H_item_length].reshape(A2H_item_length, -1)
                    
                    target_velocity = self.velocity_pose[file_index][A2H_history_start + A2H_receptive_field : A2H_history_start + A2H_item_length + 1]
                    history_velocity = self.velocity_pose[file_index][A2H_history_start : A2H_history_start + A2H_item_length].reshape(A2H_item_length, -1)
                    target_info = torch.from_numpy(np.concatenate([target_headpose, target_velocity], axis=1).reshape(current_target_length, -1)).float()
                else:
                    history_headpose = self.headposes[file_index][A2H_history_start : A2H_history_start + A2H_item_length].reshape(A2H_item_length, -1)
                    history_velocity = self.velocity_pose[file_index][A2H_history_start : A2H_history_start + A2H_item_length].reshape(A2H_item_length, -1)
                    
                    
                    target_headpose_ = self.headposes[file_index][A2H_history_start + A2H_receptive_field - self.predict_len : A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_headpose = np.zeros([current_target_length, self.predict_length, target_headpose_.shape[1]])
                    for i in range(current_target_length):
                        target_headpose[i] = target_headpose_[i : i + self.predict_length]
                    target_headpose = target_headpose#.reshape(current_target_length, -1, order='F')
                
                    target_velocity_ = self.headposes[file_index][A2H_history_start + A2H_receptive_field - self.predict_len : A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_velocity = np.zeros([current_target_length, self.predict_length, target_velocity_.shape[1]])
                    for i in range(current_target_length):
                        target_velocity[i] = target_velocity_[i : i + self.predict_length]
                    target_velocity = target_velocity#.reshape(current_target_length, -1, order='F')           
                
                    target_info = torch.from_numpy(np.concatenate([target_headpose, target_velocity], axis=2).reshape(current_target_length, -1)).float()
                
                A2Hsamples = torch.from_numpy(A2Hsamples).float()
                
    
                history_info = torch.from_numpy(np.concatenate([history_headpose, history_velocity], axis=1)).float()
            
                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, history_info, target_info
            
            
            elif self.opt.feature_decoder == 'LSTM':
                A2Hsamples = self.audio_features[file_index][current_frame * 2 : (current_frame + self.opt.A2H_receptive_field) * 2]            
                
                target_headpose = self.headposes[file_index][current_frame : current_frame + self.opt.A2H_receptive_field]      
                target_velocity = self.velocity_pose[file_index][current_frame : current_frame + self.opt.A2H_receptive_field]
                target_info = torch.from_numpy(np.concatenate([target_headpose, target_velocity], axis=1).reshape(self.opt.A2H_receptive_field, -1)).float()
                
                A2Hsamples = torch.from_numpy(A2Hsamples).float()

                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, target_info
            


    def __len__(self):        
        return self.total_len
    
         
    

    
    

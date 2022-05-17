import os
import subprocess
from os.path import join
import yaml
import tempfile
import argparse
from skimage.io import imread
import numpy as np
import librosa
from util import util
from tqdm import tqdm
import torch
from collections import OrderedDict
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from cog import BasePredictor, Input, Path
import scipy.io as sio
import albumentations as A
from options.test_audio2feature_options import TestOptions as FeatureOptions
from options.test_audio2headpose_options import TestOptions as HeadposeOptions
from options.test_feature2face_options import TestOptions as RenderOptions
from datasets import create_dataset
from models import create_model
from models.networks import APC_encoder
from util.visualizer import Visualizer
from funcs import utils, audio_funcs
from demo import write_video_with_audio
import warnings

warnings.filterwarnings("ignore")


class Predictor(BasePredictor):
    def setup(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--id', default='May', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")
        self.parser.add_argument('--driving_audio', default='data/Input/00083.wav', help="path to driving audio")
        self.parser.add_argument('--save_intermediates', default=0, help="whether to save intermediate results")

    def predict(self, 
        driving_audio: Path = Input(description='driving audio, if the file is more than 20 seconds, only the first 20 seconds will be processed for video generation'),
        talking_head: str = Input(description="choose a talking head", choices=['May', 'Obama1', 'Obama2', 'Nadella', 'McStay'], default='May')
    ) -> Path:

        ############################### I/O Settings ##############################
        # load config files
        opt = self.parser.parse_args('')
        opt.driving_audio = str(driving_audio)
        opt.id = talking_head
        with open(join('config', opt.id + '.yaml')) as f:
            config = yaml.safe_load(f)
        data_root = join('data', opt.id)

        ############################ Hyper Parameters #############################
        h, w, sr, FPS = 512, 512, 16000, 60
        mouth_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
        eye_brow_indices = [27, 65, 28, 68, 29, 67, 30, 66, 31, 72, 32, 69, 33, 70, 34, 71]
        eye_brow_indices = np.array(eye_brow_indices, np.int32)

        ############################ Pre-defined Data #############################
        mean_pts3d = np.load(join(data_root, 'mean_pts3d.npy'))
        fit_data = np.load(config['dataset_params']['fit_data_path'])
        pts3d = np.load(config['dataset_params']['pts3d_path']) - mean_pts3d
        trans = fit_data['trans'][:, :, 0].astype(np.float32)
        mean_translation = trans.mean(axis=0)
        candidate_eye_brow = pts3d[10:, eye_brow_indices]
        std_mean_pts3d = np.load(config['dataset_params']['pts3d_path']).mean(axis=0)
        # candidates images
        img_candidates = []
        for j in range(4):
            output = imread(join(data_root, 'candidates', f'normalized_full_{j}.jpg'))
            output = A.pytorch.transforms.ToTensor(normalize={'mean': (0.5, 0.5, 0.5),
                                                              'std': (0.5, 0.5, 0.5)})(image=output)['image']
            img_candidates.append(output)
        img_candidates = torch.cat(img_candidates).unsqueeze(0).cuda()

        # shoulders
        shoulders = np.load(join(data_root, 'normalized_shoulder_points.npy'))
        shoulder3D = np.load(join(data_root, 'shoulder_points3D.npy'))[1]
        ref_trans = trans[1]

        # camera matrix, we always use training set intrinsic parameters.
        camera = utils.camera()
        camera_intrinsic = np.load(join(data_root, 'camera_intrinsic.npy')).astype(np.float32)
        APC_feat_database = np.load(join(data_root, 'APC_feature_base.npy'))

        # load reconstruction data
        scale = sio.loadmat(join(data_root, 'id_scale.mat'))['scale'][0, 0]
        Audio2Mel_torch = audio_funcs.Audio2Mel(n_fft=512, hop_length=int(16000 / 120), win_length=int(16000 / 60),
                                                sampling_rate=16000,
                                                n_mel_channels=80, mel_fmin=90, mel_fmax=7600.0).cuda()

        ########################### Experiment Settings ###########################
        #### user config
        use_LLE = config['model_params']['APC']['use_LLE']
        Knear = config['model_params']['APC']['Knear']
        LLE_percent = config['model_params']['APC']['LLE_percent']
        headpose_sigma = config['model_params']['Headpose']['sigma']
        Feat_smooth_sigma = config['model_params']['Audio2Mouth']['smooth']
        Head_smooth_sigma = config['model_params']['Headpose']['smooth']
        Feat_center_smooth_sigma, Head_center_smooth_sigma = 0, 0
        AMP_method = config['model_params']['Audio2Mouth']['AMP'][0]
        Feat_AMPs = config['model_params']['Audio2Mouth']['AMP'][1:]
        rot_AMP, trans_AMP = config['model_params']['Headpose']['AMP']
        shoulder_AMP = config['model_params']['Headpose']['shoulder_AMP']
        save_feature_maps = config['model_params']['Image2Image']['save_input']

        #### common settings
        Featopt = FeatureOptions().parse()
        Headopt = HeadposeOptions().parse()
        Renderopt = RenderOptions().parse()
        Featopt.load_epoch = config['model_params']['Audio2Mouth']['ckp_path']
        Headopt.load_epoch = config['model_params']['Headpose']['ckp_path']
        Renderopt.dataroot = config['dataset_params']['root']
        Renderopt.load_epoch = config['model_params']['Image2Image']['ckp_path']
        Renderopt.size = config['model_params']['Image2Image']['size']

        ############################# Load Models #################################
        print('---------- Loading Model: APC-------------')
        APC_model = APC_encoder(config['model_params']['APC']['mel_dim'],
                                config['model_params']['APC']['hidden_size'],
                                config['model_params']['APC']['num_layers'],
                                config['model_params']['APC']['residual'])
        # load all 5 here?
        APC_model.load_state_dict(torch.load(config['model_params']['APC']['ckp_path']), strict=False)
        APC_model.cuda()
        APC_model.eval()
        print('---------- Loading Model: {} -------------'.format(Featopt.task))
        Audio2Feature = create_model(Featopt)
        Audio2Feature.setup(Featopt)
        Audio2Feature.eval()
        print('---------- Loading Model: {} -------------'.format(Headopt.task))
        Audio2Headpose = create_model(Headopt)
        Audio2Headpose.setup(Headopt)
        Audio2Headpose.eval()
        if Headopt.feature_decoder == 'WaveNet':
            Headopt.A2H_receptive_field = Audio2Headpose.Audio2Headpose.module.WaveNet.receptive_field
        print('---------- Loading Model: {} -------------'.format(Renderopt.task))
        facedataset = create_dataset(Renderopt)
        Feature2Face = create_model(Renderopt)
        Feature2Face.setup(Renderopt)
        Feature2Face.eval()
        visualizer = Visualizer(Renderopt)

        # check audio duration and trim audio
        extension_name = os.path.basename(opt.driving_audio).split('.')[-1]
        audio_threshold = 10
        duration = librosa.get_duration(filename=opt.driving_audio)
        if duration > audio_threshold:
            print(f'audio file is longer than {audio_threshold} seconds, trimming the first {audio_threshold} seconds '
                  f'for further processing')
            ffmpeg_extract_subclip(opt.driving_audio, 0, audio_threshold, targetname=f'shorter_input.{extension_name}')
            opt.driving_audio = f'shorter_input.{extension_name}'

        # create the results folder
        audio_name = os.path.basename(opt.driving_audio).split('.')[0]
        save_root = join('results', opt.id, audio_name)
        os.makedirs(save_root, exist_ok=True)
        clean_folder(save_root)
        out_path = Path(tempfile.mkdtemp()) / "out.mp4"

        ############################## Inference ##################################
        print('Processing audio: {} ...'.format(audio_name))
        # read audio
        audio, _ = librosa.load(opt.driving_audio, sr=sr)
        total_frames = np.int32(audio.shape[0] / sr * FPS)

        #### 1. compute APC features
        print('1. Computing APC features...')
        mel80 = utils.compute_mel_one_sequence(audio)
        mel_nframe = mel80.shape[0]
        with torch.no_grad():
            length = torch.Tensor([mel_nframe])
            mel80_torch = torch.from_numpy(mel80.astype(np.float32)).cuda().unsqueeze(0)
            hidden_reps = APC_model.forward(mel80_torch, length)[0]  # [mel_nframe, 512]
            hidden_reps = hidden_reps.cpu().numpy()
        audio_feats = hidden_reps

        #### 2. manifold projection
        if use_LLE:
            print('2. Manifold projection...')
            ind = utils.KNN_with_torch(audio_feats, APC_feat_database, K=Knear)
            weights, feat_fuse = utils.compute_LLE_projection_all_frame(audio_feats, APC_feat_database, ind,
                                                                        audio_feats.shape[0])
            audio_feats = audio_feats * (1 - LLE_percent) + feat_fuse * LLE_percent

        #### 3. Audio2Mouth
        print('3. Audio2Mouth inference...')
        pred_Feat = Audio2Feature.generate_sequences(audio_feats, sr, FPS, fill_zero=True, opt=Featopt)

        #### 4. Audio2Headpose
        print('4. Headpose inference...')
        # set history headposes as zero
        pre_headpose = np.zeros(Headopt.A2H_wavenet_input_channels, np.float32)
        pred_Head = Audio2Headpose.generate_sequences(audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.3,
                                                      opt=Headopt)

        #### 5. Post-Processing
        print('5. Post-processing...')
        nframe = min(pred_Feat.shape[0], pred_Head.shape[0])
        pred_pts3d = np.zeros([nframe, 73, 3])
        pred_pts3d[:, mouth_indices] = pred_Feat.reshape(-1, 25, 3)[:nframe]

        ## mouth
        pred_pts3d = utils.landmark_smooth_3d(pred_pts3d, Feat_smooth_sigma, area='only_mouth')
        pred_pts3d = utils.mouth_pts_AMP(pred_pts3d, True, AMP_method, Feat_AMPs)
        pred_pts3d = pred_pts3d + mean_pts3d
        pred_pts3d = utils.solve_intersect_mouth(pred_pts3d)  # solve intersect lips if exist

        ## headpose
        pred_Head[:, 0:3] *= rot_AMP
        pred_Head[:, 3:6] *= trans_AMP
        pred_headpose = utils.headpose_smooth(pred_Head[:, :6], Head_smooth_sigma).astype(np.float32)
        pred_headpose[:, 3:] += mean_translation
        pred_headpose[:, 0] += 180

        ## compute projected landmarks
        pred_landmarks = np.zeros([nframe, 73, 2], dtype=np.float32)
        final_pts3d = np.zeros([nframe, 73, 3], dtype=np.float32)
        final_pts3d[:] = std_mean_pts3d.copy()
        final_pts3d[:, 46:64] = pred_pts3d[:nframe, 46:64]
        for k in tqdm(range(nframe)):
            ind = k % candidate_eye_brow.shape[0]
            final_pts3d[k, eye_brow_indices] = candidate_eye_brow[ind] + mean_pts3d[eye_brow_indices]
            pred_landmarks[k], _, _ = utils.project_landmarks(camera_intrinsic, camera.relative_rotation,
                                                              camera.relative_translation, scale,
                                                              pred_headpose[k], final_pts3d[k])

            ## Upper Body Motion
        pred_shoulders = np.zeros([nframe, 18, 2], dtype=np.float32)
        pred_shoulders3D = np.zeros([nframe, 18, 3], dtype=np.float32)
        for k in range(nframe):
            diff_trans = pred_headpose[k][3:] - ref_trans
            pred_shoulders3D[k] = shoulder3D + diff_trans * shoulder_AMP
            # project
            project = camera_intrinsic.dot(pred_shoulders3D[k].T)
            project[:2, :] /= project[2, :]  # divide z
            pred_shoulders[k] = project[:2, :].T

        #### 6. Image2Image translation & Save resuls
        print('6. Image2Image translation & Saving results...')
        for ind in tqdm(range(0, nframe), desc='Image2Image translation inference'):
            # feature_map: [input_nc, h, w]
            current_pred_feature_map = facedataset.dataset.get_data_test_mode(pred_landmarks[ind],
                                                                              pred_shoulders[ind],
                                                                              facedataset.dataset.image_pad)
            input_feature_maps = current_pred_feature_map.unsqueeze(0).cuda()
            pred_fake = Feature2Face.inference(input_feature_maps, img_candidates)
            # save results
            visual_list = [('pred', util.tensor2im(pred_fake[0]))]
            if save_feature_maps:
                visual_list += [('input', np.uint8(current_pred_feature_map[0].cpu().numpy() * 255))]
            visuals = OrderedDict(visual_list)
            visualizer.save_images(save_root, visuals, str(ind + 1))

        ## make videos
        # generate corresponding audio, reused for all results
        tmp_audio_path = join(save_root, 'tmp.wav')
        tmp_audio_clip = audio[: np.int32(nframe * sr / FPS)]
        librosa.output.write_wav(tmp_audio_path, tmp_audio_clip, sr)

        def write_video_with_audio(audio_path, output_path, prefix='pred_'):
            fps, fourcc = 60, cv2.VideoWriter_fourcc(*'DIVX')
            video_tmp_path = join(save_root, 'tmp.avi')
            out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (Renderopt.loadSize, Renderopt.loadSize))
            for j in tqdm(range(nframe), position=0, desc='writing video'):
                img = cv2.imread(join(save_root, prefix + str(j + 1) + '.jpg'))
                out.write(img)
            out.release()
            cmd = 'ffmpeg -i "' + video_tmp_path + '" -i "' + audio_path + '" -codec copy -shortest "' + output_path + '"'
            subprocess.call(cmd, shell=True)
            os.remove(video_tmp_path)  # remove the template video

        temp_out = 'temp_video.avi'
        write_video_with_audio(tmp_audio_path, temp_out, 'pred_')
        # convert to mp4
        cmd = ("ffmpeg -i "
               + temp_out + " -strict -2 "
               + str(out_path)
               )
        subprocess.call(cmd, shell=True)

        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        if os.path.exists(temp_out):
            os.remove(temp_out)
        if os.path.exists(f'shorter_input.{extension_name}'):
            os.remove(f'shorter_input.{extension_name}')
        if not opt.save_intermediates:
            _img_paths = list(map(lambda x: str(x), list(Path(save_root).glob('*.jpg'))))
            for i in tqdm(range(len(_img_paths)), desc='deleting intermediate images'):
                os.remove(_img_paths[i])

        print('Finish!')

        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
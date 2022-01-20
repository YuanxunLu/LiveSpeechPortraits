import os
from datasets.base_dataset import BaseDataset
import os.path
from pathlib import Path
import torch
from skimage.io import imread, imsave
from PIL import Image
import bisect
import numpy as np
import io
import cv2
import h5py
import albumentations as A


class FaceDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.state = 'Train' if self.opt.isTrain else 'Test'
        self.dataset_name = opt.dataset_names[0]
        
        # default settings
        # currently, we have 8 parts for face parts
        self.part_list = [[list(range(0, 15))],                                # contour
                          [[15,16,17,18,18,19,20,15]],                         # right eyebrow
                          [[21,22,23,24,24,25,26,21]],                         # left eyebrow
                          [range(35, 44)],                                     # nose
                          [[27,65,28,68,29], [29,67,30,66,27]],                # right eye
                          [[33,69,32,72,31], [31,71,34,70,33]],                # left eye
                          [range(46, 53), [52,53,54,55,56,57,46]],             # mouth
                          [[46,63,62,61,52], [52,60,59,58,46]]                 # tongue
                         ]
        self.mouth_outer = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 46]
        self.label_list = [1, 1, 2, 3, 3, 4, 5] # labeling for different facial parts
                
        # only load in train mode
          
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        if self.state == 'Train':
            self.clip_names = opt.train_dataset_names
        elif self.state == 'Val':
            self.clip_names = opt.validate_dataset_names
        elif self.state == 'Test':
            self.clip_names = opt.test_dataset_names
        
        self.clip_nums = len(self.clip_names)
        
        # load pts & image info
        self.landmarks2D, self.len, self.sample_len = [''] * self.clip_nums, [''] * self.clip_nums, [''] * self.clip_nums
        self.image_transforms, self.image_pad, self.tgts_paths = [''] * self.clip_nums, [''] * self.clip_nums, [''] * self.clip_nums
        self.shoulders, self.shoulder3D = [''] * self.clip_nums, [''] * self.clip_nums
        self.sample_start = []
        
        # tracked 3d info & candidates images
        self.pts3d, self.rot, self.trans = [''] * self.clip_nums, [''] * self.clip_nums, [''] * self.clip_nums
        self.full_cand = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums
        
        self.total_len = 0
        if self.opt.isTrain:
            for i in range(self.clip_nums):
                name = self.clip_names[i]
                clip_root = os.path.join(self.dataset_root, name)
                # basic image info
                img_file_path = os.path.join(clip_root, name + '.h5')
                img_file = h5py.File(img_file_path, 'r')[name]
                example = np.asarray(Image.open(io.BytesIO(img_file[0]))) 
                h, w, _ = example.shape
                
                
                landmark_path = os.path.join(clip_root, 'tracked2D_normalized_pts_fix_contour.npy')
                self.landmarks2D[i] = np.load(landmark_path).astype(np.float32)
                change_paras = np.load(os.path.join(clip_root, 'change_paras.npz'))
                scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
                x_min, x_max, y_min, y_max = xc-256, xc+256, yc-256, yc+256
                # if need padding
                x_min, x_max, y_min, y_max, self.image_pad[i] = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max, h), None
     
                
                if x_min == 0 or x_max == 512 or y_min == 0 or y_max == 512:
                    top, bottom, left, right = abs(yc-256-y_min), abs(yc+256-y_max), abs(xc-256-x_min), abs(xc+256-x_max)
                    self.image_pad[i] = [top, bottom, left, right]      
                self.image_transforms[i] = A.Compose([
                        A.Resize(np.int32(h*scale), np.int32(w*scale)),
                        A.Crop(x_min, y_min, x_max, y_max)])
                
                if self.opt.isH5:
                    tgt_file_path = os.path.join(clip_root, name + '.h5')
                    tgt_file = h5py.File(tgt_file_path, 'r')[name]
                    image_length = len(tgt_file)
                else:
                    tgt_paths = list(map(lambda x:str(x), sorted(list(Path(clip_root).glob('*'+self.opt.suffix)), key=lambda x: int(x.stem))))
                    image_length = len(tgt_paths)
                    self.tgts_paths[i] = tgt_paths
                if not self.landmarks2D[i].shape[0] == image_length:
                    raise ValueError('In dataset {} length of landmarks and images are not equal!'.format(name))
                      
                # tracked 3d info 
                fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
                fit_data = np.load(fit_data_path)
                self.pts3d[i] = fit_data['pts_3d'].astype(np.float32)
                self.rot[i] = fit_data['rot_angles'].astype(np.float32)
                self.trans[i] = fit_data['trans'][:,:,0].astype(np.float32)
                if not self.pts3d[i].shape[0] == image_length:
                    raise ValueError('In dataset {} length of 3d pts and images are not equal!'.format(name))  
                    
                # candidates images
    
                tmp = []
                for j in range(4):
                    try:
                        output = imread(os.path.join(clip_root, 'candidates', f'normalized_full_{j}.jpg'))
                    except:
                        imgc = imread(os.path.join(clip_root, 'candidates', f'full_{j}.jpg'))
                        output = self.common_dataset_transform(imgc, i)
                        imsave(os.path.join(clip_root, 'candidates', f'normalized_full_{j}.jpg'), output)
                    output = A.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
                    tmp.append(output)
                self.full_cand[i] = torch.cat(tmp)             
                
                # headpose
                fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
                fit_data = np.load(fit_data_path)
                rot_angles = fit_data['rot_angles'].astype(np.float32)
                # change -180~180 to 0~360
                if not self.dataset_name  == 'Yuxuan':
                    rot_change = rot_angles[:, 0] < 0
                    rot_angles[rot_change, 0] += 360
                    rot_angles[:,0] -= 180   # change x axis direction
                # use delta translation
                mean_trans = fit_data['trans'][:,:,0].astype(np.float32).mean(axis=0)
                trans = fit_data['trans'][:,:,0].astype(np.float32) - mean_trans
                
                self.headposes[i] = np.concatenate([rot_angles, trans], axis=1)
                
                # shoulders
                shoulder_path = os.path.join(clip_root, 'normalized_shoulder_points.npy')
                self.shoulders[i] = np.load(shoulder_path)
                shoulder3D_path = os.path.join(clip_root, 'shoulder_points3D.npy')
                self.shoulder3D[i] = np.load(shoulder3D_path)
                    
                                               
                self.sample_len[i] = np.int32(np.floor((self.landmarks2D[i].shape[0] - 60) / self.opt.frame_jump) + 1)
                self.len[i] = self.landmarks2D[i].shape[0]
                if i == 0:
                    self.sample_start.append(0)
                else:
                    self.sample_start.append(self.sample_start[-1] + self.sample_len[i-1])  # not minus 1
                self.total_len += self.sample_len[i]
        
        # test mode        
        else:
            # if need padding
            example = imread(os.path.join(self.root, 'example.png'))
            h, w, _ = example.shape
            change_paras = np.load(os.path.join(self.root, 'change_paras.npz'))
            scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
            x_min, x_max, y_min, y_max = xc-256, xc+256, yc-256, yc+256
            x_min, x_max, y_min, y_max, self.image_pad = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max, h), None
 
            
            if x_min == 0 or x_max == 512 or y_min == 0 or y_max == 512:
                top, bottom, left, right = abs(yc-256-y_min), abs(yc+256-y_max), abs(xc-256-x_min), abs(xc+256-x_max)
                self.image_pad = [top, bottom, left, right]
            
                
  
    
    
    def __getitem__(self, ind):
        dataset_index = bisect.bisect_right(self.sample_start, ind) - 1
        data_index = (ind - self.sample_start[dataset_index]) * self.opt.frame_jump + np.random.randint(self.opt.frame_jump)
        
        target_ind = data_index + 1  # history_ind, current_ind
        landmarks = self.landmarks2D[dataset_index][target_ind]  # [73, 2]
        shoulders = self.shoulders[dataset_index][target_ind].copy()
        
        dataset_name = self.clip_names[dataset_index]
        clip_root = os.path.join(self.dataset_root, dataset_name)
        if self.opt.isH5:
            tgt_file_path = os.path.join(clip_root, dataset_name + '.h5')
            tgt_file = h5py.File(tgt_file_path, 'r')[dataset_name]
            tgt_image = np.asarray(Image.open(io.BytesIO(tgt_file[target_ind]))) 
                        
            # do transform
            tgt_image = self.common_dataset_transform(tgt_image, dataset_index, None)
        else:
            pass

        h, w, _ = tgt_image.shape
        
        ### transformations & online data augmentations on images and landmarks   
        self.get_crop_coords(landmarks, (w, h), dataset_name, random_trans_scale=0)  # 30.5 µs ± 348 ns  random translation    
              
        transform_tgt = self.get_transform(dataset_name, True, n_img=1, n_keypoint=1, flip=False) 
        transformed_tgt = transform_tgt(image=tgt_image, keypoints=landmarks)        
                 
        tgt_image, points = transformed_tgt['image'], np.array(transformed_tgt['keypoints']).astype(np.float32)
        
        feature_map = self.get_feature_image(points, (self.opt.loadSize, self.opt.loadSize), shoulders, self.image_pad[dataset_index])[np.newaxis, :].astype(np.float32)/255.
        feature_map = torch.from_numpy(feature_map)
        
        ## facial weight mask
        weight_mask = self.generate_facial_weight_mask(points, h, w)[None, :]
        
        cand_image = self.full_cand[dataset_index]
        
        return_list = {'feature_map': feature_map, 'cand_image': cand_image, 'tgt_image': tgt_image, 'weight_mask': weight_mask}
           
        return return_list
   
    
    
    
    def common_dataset_transform(self, input, i):
        output = self.image_transforms[i](image=input)['image']
        if self.image_pad[i] is not None:
            top, bottom, left, right = self.image_pad[i]
            output = cv2.copyMakeBorder(output, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)
        return output
    
    
    
    def generate_facial_weight_mask(self, points, h = 512, w = 512):
        mouth_mask = np.zeros([512, 512, 1])
        points = points[self.mouth_outer]
        points = np.int32(points)
        mouth_mask = cv2.fillPoly(mouth_mask, [points], (255,0,0))
#        plt.imshow(mouth_mask[:,:,0])
        mouth_mask = cv2.dilate(mouth_mask, np.ones((45, 45))) / 255
        
        return mouth_mask.astype(np.float32)
    
      
    
    def get_transform(self, dataset_name, keypoints=False, n_img=1, n_keypoint=1, normalize=True, flip=False):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))
                
        additional_flag = False
        additional_targets_dict = {}
        if n_img > 1:
            additional_flag = True
            image_str = ['image' + str(i) for i in range(0, n_img)]
            for i in range(n_img):
                additional_targets_dict[image_str[i]] = 'image'
        if n_keypoint > 1:
            additional_flag = True
            keypoint_str = ['keypoint' + str(i) for i in range(0, n_keypoint)]
            for i in range(n_keypoint):
                additional_targets_dict[keypoint_str[i]] = 'keypoints'
        
        transform = A.Compose([
                A.Crop(x_min=min_x, x_max=max_x, y_min=min_y, y_max=max_y),
                A.Resize(self.opt.loadSize, self.opt.loadSize),
                A.HorizontalFlip(p=flip),
                A.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)} if normalize==True else None)],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) if keypoints==True else None,
                additional_targets=additional_targets_dict if additional_flag == True else None
                )
        return transform        
     
    
    def get_data_test_mode(self, landmarks, shoulder, pad=None):
        ''' get transformed data
        '''
       
        feature_map = torch.from_numpy(self.get_feature_image(landmarks, (self.opt.loadSize, self.opt.loadSize), shoulder, pad)[np.newaxis, :].astype(np.float32)/255.)

        return feature_map  


    def get_feature_image(self, landmarks, size, shoulders=None, image_pad=None):
        # draw edges
        im_edges = self.draw_face_feature_maps(landmarks, size)  
        if shoulders is not None:
            if image_pad is not None:
                top, bottom, left, right = image_pad
                delta_y = top - bottom
                delta_x = right - left
                shoulders[:, 0] += delta_x
                shoulders[:, 1] += delta_y
            im_edges = self.draw_shoulder_points(im_edges, shoulders)
        

        return im_edges


    def draw_shoulder_points(self, img, shoulder_points):
        num = int(shoulder_points.shape[0] / 2)
        for i in range(2):
            for j in range(num - 1):
                pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                img = cv2.line(img, tuple(pt1), tuple(pt2), 255, 2)  # BGR
        
        return img

    
    def draw_face_feature_maps(self, keypoints, size=(512, 512)):
        w, h = size
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(len(edge)-1):
                    pt1 = [int(flt) for flt in keypoints[edge[i]]]
                    pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                    im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)

        return im_edges


    def get_crop_coords(self, keypoints, size, dataset_name, random_trans_scale=50): 
        # cut a rought region for fine cutting
        # here x towards right and y towards down, origin is left-up corner
        w_ori, h_ori = size             
        min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
        min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()                
        xc = (min_x + max_x) // 2
        yc = (min_y*3 + max_y) // 4
        h = w = min((max_x - min_x) * 2, w_ori, h_ori)
        
        if self.opt.isTrain:
            # do online augment on landmarks & images
            # 1. random translation: move 10%
            x_bias, y_bias = np.random.uniform(-random_trans_scale, random_trans_scale, size=(2,))
            xc, yc = xc + x_bias, yc + y_bias
            
        # modify the center x, center y to valid position
        xc = min(max(0, xc - w//2) + w, w_ori) - w//2
        yc = min(max(0, yc - h//2) + h, h_ori) - h//2
        
        min_x, max_x = xc - w//2, xc + w//2
        min_y, max_y = yc - h//2, yc + h//2 
        
        setattr(self, 'min_x_' + str(dataset_name), int(min_x))
        setattr(self, 'max_x_' + str(dataset_name), int(max_x))
        setattr(self, 'min_y_' + str(dataset_name), int(min_y))
        setattr(self, 'max_y_' + str(dataset_name), int(max_y))


    def crop(self, img, dataset_name):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))    
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))
  

    def __len__(self):  
        if self.opt.isTrain:
            return self.total_len
        else:
            return 1

    def name(self):
        return 'FaceDataset'
    


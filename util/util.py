from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from collections import OrderedDict

from . import flow_viz



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def tensor2flow(flo, imtype=np.uint8):
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


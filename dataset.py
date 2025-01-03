import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os

def normalize_numpy(data):
    m = 0.5
    std = 0.5
    return (data - m)/std

def rggb_pack_aveg(im):
    im = (im*255).astype('uint8')
    im = im/255.
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]   
    ## r g2 g1 b
    out = np.concatenate((im[0:H:2,0:W:2,:],  #r
                          (im[0:H:2,1:W:2,:]+im[1:H:2,0:W:2,:])/2,     #(g1+g2)/2
                          im[1:H:2,1:W:2,:]), axis=2)  #b
    return out


def pack_rggb_raw(im):
    #pack RGGB Bayer raw to 4 channels

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out


def default_loader(path,type_img):
    if type_img == 'm_raw':
        raw_img = np.load(path)
        raw_data = raw_img['patch_data']
        #norm_factor = raw_img['white_level'] - raw_img['black_level_per_channel'][0]
        #img = (raw_data- raw_img['black_level_per_channel'][0])/norm_factor
        img = raw_data/4095.0

    elif type_img == 'rgb':
        img = np.array(Image.open(path).convert('RGB'))/255.0
        
    elif type_img == 'gt_raw':
        raw_img = np.load(path)
        raw_data = raw_img['patch_data']
        img = raw_data/4095.0

    return img

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    # im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')

class Moire_dataset_test(Dataset):
    def __init__(self, root, loader = default_loader):
        moire_raw_root = os.path.join(root, 'moire_RAW_npz')
        gt_raw_root = os.path.join(root, 'gt_RAW_npz')
        gt_rgb_root = os.path.join(root, 'gt_RGB')

        image_names = os.listdir(moire_raw_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_raw_images = [os.path.join(moire_raw_root, x + '_m.npz') for x in image_names]
        self.gt_raw_images = [os.path.join(gt_raw_root, x + '_gt.npz') for x in image_names]
        self.gt_rgb_images = [os.path.join(gt_rgb_root, x + '_gt.png') for x in image_names]
        
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_raw_path = self.moire_raw_images[index]
        gt_rgb_path = self.gt_rgb_images[index]
        gt_raw_path = self.gt_raw_images[index]
        moire_raw_img = self.loader(moire_raw_path, 'm_raw')
        gt_raw_img = self.loader(gt_raw_path, 'gt_raw')
        gt_rgb_img = self.loader(gt_rgb_path, 'rgb')

        class_img = rggb_pack_aveg(moire_raw_img)
        class_img = normalize_numpy(class_img)
        moire_raw_img = pack_rggb_raw(moire_raw_img)	
        gt_raw_img = pack_rggb_raw(gt_raw_img)

        moire_raw_img = torch.from_numpy(moire_raw_img)
        gt_raw_img = torch.from_numpy(gt_raw_img)
        class_img = torch.from_numpy(class_img)
        gt_rgb_img = torch.from_numpy(gt_rgb_img)
        
        moire_raw_img = moire_raw_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        gt_raw_img = gt_raw_img.type(torch.FloatTensor).permute(2,0,1).cuda()
        class_img = class_img.type(torch.FloatTensor).permute(2,0,1).cuda()
        gt_rgb_img = gt_rgb_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        num_img = self.labels[index]
        
        return moire_raw_img,gt_raw_img, class_img, gt_rgb_img, num_img#, label

    def __len__(self):
        return len(self.moire_raw_images)



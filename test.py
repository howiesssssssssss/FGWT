from __future__ import division
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import Moire_dataset_test,save_images
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--checkpoint_dir',  default='', help='the model file to load')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--num_worker', dest='num_worker', type=int, default=0, help='number of workers when loading data')
parser.add_argument('--test_path', dest='test_path', default='', help='path of test data')
parser.add_argument('--save_test_dir', dest='save_test_dir', default='./out/', help='storage path of output data')
parser.add_argument('-v','--version', type=str, default='0.1')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if not os.path.exists(args.save_test_dir):
    os.makedirs(args.save_test_dir)


Moire_data_test = Moire_dataset_test(args.test_path)
test_dataloader = DataLoader(Moire_data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_worker,
                             drop_last=False)

model = torch.load(args.checkpoint_dir).cuda()


loss_fn = lpips.LPIPS(net='alex').cuda()
model.eval()

psnr = []
ssim = []
lpip = []

with torch.no_grad():
    for i ,( moire_raw_img,gt_raw_img, class_img,moire_rgb_img, gt_rgb_img, num_img) in enumerate(test_dataloader):      

        dm_rgb = model(moire_rgb_img)

        pre = torch.clamp(dm_rgb, min=0, max=1)
        tar = torch.clamp(gt_rgb_img, min=0, max=1)
        res_lpips = loss_fn.forward(pre, tar, normalize=True).item()

        psnr_output = PSNR(torch.clamp(dm_rgb,0,1).cpu().numpy(), gt_rgb_img.cpu().numpy())
        ssim_output = SSIM(torch.clamp(dm_rgb,0,1).squeeze().cpu().numpy(), gt_rgb_img.squeeze().cpu().numpy(),channel_axis=0,data_range=1)

        psnr.append(psnr_output)
        ssim.append(ssim_output)
        lpip.append(res_lpips)
        print("[*] Testing: the PSNR is "+str(psnr_output)+" the SSIM is "+str(ssim_output)+" the LPIPS is "+str(res_lpips))

        # ##save image
        dm_rgb = dm_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
        dm_rgb = np.clip(255 * dm_rgb, 0, 255).astype('uint8')
        save_images(os.path.join(args.save_test_dir,str(num_img[0])+'_dm.png'), dm_rgb)

    psnr_mean = np.mean(psnr)
    ssim_mean = np.mean(ssim)
    lpip_mean = np.mean(lpip)

    print("[*] the average of PSNR is "+str(psnr_mean))  
    print("[*] the average of SSIM is "+str(ssim_mean))  
    print("[*] the average of LPIPS is "+str(lpip_mean))
    
    print("[*] Finish testing.")

torch.cuda.empty_cache () 


  

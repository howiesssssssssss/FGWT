from __future__ import division
import os, scipy.io
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--checkpoint_dir',  default='', help='the model file to load')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_worker', dest='num_worker', type=int, default=0, help='number of workers when loading data')
parser.add_argument('--test_path', dest='test_path', default='', help='path of test data')
parser.add_argument('--save_test_dir', dest='save_test_dir', default='./out_final/', help='storage path of output data')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if not os.path.exists(args.save_test_dir):
    os.makedirs(args.save_test_dir)


Moire_data_test = Moire_dataset_test(args.test_path)
test_dataloader = DataLoader(Moire_data_test,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False)


model = torch.load(args.checkpoint_dir).cuda()
model_isp = torch.load('ISP_model/isp_model_epoch20.pth').cuda()


model.eval()
model_isp.eval()
	
psnr = []
ssim = []
psnrraw = []
ssimraw = []



with torch.no_grad():
    for ii ,(moire_img,gt_raw, c_img, gt_img, num_img) in enumerate(test_dataloader):      
        
        dm_raw = model(moire_img)
        
        dm_rgb = model_isp(dm_raw)

        ##Rgb
        # psnr_output = PSNR(torch.clamp(dm_rgb,0,1).cpu().numpy(), gt_img.cpu().numpy())
        # ssim_output = SSIM(torch.clamp(dm_rgb,0,1).squeeze().cpu().numpy(), gt_img.squeeze().cpu().numpy(),channel_axis=0,data_range=1)
        # psnr.append(psnr_output)
        # ssim.append(ssim_output)

        # print("RGB_num_img = "+str(num_img)+" PSNR = "+str(psnr_output))
        # #print("num_img = "+str(num_img)+" SSIM = "+str(ssim_output1))
        # psnr_txt1="RGB_num_img = "+str(num_img)+" PSNR = "+str(psnr_output)
        # psnr_txt2="RGB_num_img = "+str(num_img)+" SSIM = "+str(ssim_output)

        ##RAW
        psnr_output1 = PSNR(torch.clamp(dm_raw,0,1).cpu().numpy(), gt_raw.cpu().numpy())
        ssim_output1 = SSIM(torch.clamp(dm_raw,0,1).squeeze().cpu().numpy(), gt_raw.squeeze().cpu().numpy(),channel_axis=0,data_range=1)
        psnrraw.append(psnr_output1)
        ssimraw.append(ssim_output1)

        print("RAW_num_img = "+str(num_img)+" PSNR = "+str(psnr_output1))
        #print("num_img = "+str(num_img)+" SSIM = "+str(ssim_output1))
        psnr_txt3="RAW_num_img = "+str(num_img)+" PSNR = "+str(psnr_output1)
        psnr_txt4="RAW_num_img = "+str(num_img)+" SSIM = "+str(ssim_output1)

        with open(args.save_test_dir+'psnr.txt','a') as psnr_file:   
            # psnr_file.write(psnr_txt1)
            # psnr_file.write(psnr_txt2)
            psnr_file.write(psnr_txt3)
            psnr_file.write(psnr_txt4)    
            psnr_file.write('\n') 


        # ##save image			
        # dm_rgb = dm_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
        # dm_rgb = np.clip(255 * dm_rgb, 0, 255).astype('uint8')
        # save_images(os.path.join(args.save_test_dir,str(num_img[0])+'_'+str(psnr_output)[0:6]+'_dm.png'), dm_rgb)
        
    # psnr_mean = np.mean(psnr)
    # ssim_mean = np.mean(ssim)

    psnr_mean1 = np.mean(psnrraw)
    ssim_mean1 = np.mean(ssimraw)

    # print("[*] the average of RGB PSNR is "+str(psnr_mean))  
    # print("[*] the average of RGB SSIM is "+str(ssim_mean))  
    print("[*] the average of RAW PSNR is "+str(psnr_mean1))  
    print("[*] the average of RAW SSIM is "+str(ssim_mean1))
    
    with open(args.save_test_dir+'psnr.txt','a') as psnr_file: 
        # psnr_file.write("Average RGB PSNR = "+str(psnr_mean))
        # psnr_file.write("Average RGB SSIM = "+str(ssim_mean))
        psnr_file.write("Average RAW PSNR = "+str(psnr_mean1))
        psnr_file.write("Average RAW SSIM = "+str(ssim_mean1))
    
    print("[*] Finish testing.")

torch.cuda.empty_cache()

  

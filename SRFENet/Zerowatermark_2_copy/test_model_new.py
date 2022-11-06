import torch
import torch.nn
import argparse
import os
import numpy as np
from options import *
import cv2 as cv
import math
import utils
# import tqdm
from tqdm import *

from model.hidden import *
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from noise_layers.formal_noise import *
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.gaussian import Gaussian_blur
from noise_layers.jpeg import Jpeg
from noise_layers.salt_and_pepper import Salt_and_Pepper
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.Median_filter import Median_filter
from noise_layers.Adjust import *
from noise_layers.grid_crop import grid_crop
from noise_layers.resize import Resize
from noise_layers.Painting import Painting
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
from model.myplot import *
from model.my_result import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def my_centercrop(input_img, new_height, new_width):
    # input_img:PIL
    # output_img:PIL

    width, height = input_img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = input_img.crop((left, top, right, bottom))
    return image

def PSNR(img1,img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def yuv_psnr(img):
    imgy = 0.299 * img[:,0, :, :] + 0.587 * img[:, 1, :,:] + 0.114 * img[ :,2:, :,:]
    imgu = -0.14713 * img[:,0, :, :] + (-0.28886) * img[:, 1, :,:] + 0.436 * img[ :,2:, :,:]
    imgv = 0.615 * img[:,0, :, :] + -0.51499 * img[:, 1, :,:] + (-0.10001) * img[ :,2:, :,:]
    return imgy, imgu, imgv

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source_images', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument("--noise",'-n',nargs="*",action=NoiseArgParser)
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()
    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    # print("Trained_Attack:",noise_config)
    noise_config=args.noise     #Test attack:Crop()
    # print("Test_Attacked:",noise_config)
    # noiser = Noiser(noise_config,device,False)    #Noiser()
    # print(noiser)
    # noiser=Dropout([0.1,0.1])
    save_list=[]
    Attack_list=[
        # {'func':Identity,'name':'Identity','params':np.array([]),'active':True,'num_par':1},
        # {'func': New_Jpeg, 'name': 'New_Jpeg', 'params': np.array([30, 50, 70]), 'active': True, 'num_par': 1},
        {'func':Hybrid_Attack,'name':'Hybrid_Attack','params':np.array([1]),'active':True,'num_par':1},
        # {'func': Jpeg, 'name': 'Jpeg', 'params': np.array([10,30,50,70,90]), 'active':True, 'num_par': 1},
        # {'func':Gaussian_blur, 'name': 'Gauss_Blur', 'params': np.array([3, 5, 7,9]), 'std': np.array([1.0]),'active': False, 'num_par': -1},
        {'func': New_Gauss_Blur, 'name': 'New_Gauss_Blur', 'params': np.array([3, 5, 7, 9]), 'std': np.array([1.0]),
         'active': False, 'num_par': -1},
        {'func':Smooth,'name':'smooth','params':np.array([3,5,7,9]),'active':False,'num_par':1},
        {'func':Median_filter,'name':'median_filter','params':np.array([3,5,7,9]),'active':False,'num_par':1},
        # {'func':Winner,'name':'winner_filter','params':np.array([3,5,7,9]),'active':False,'num_par':1},
        {'func': Salt_and_Pepper, 'name': 'salt and pepper', 'params': np.array([0.01,0.02,0.03,0.05]),
         'active': False, 'num_par': 1},
        {'func': Gaussian_Noise, 'name': 'Gauss_Noise',
         'params': np.array([0.03,0.045,0.07,0.1,0.141,0.22]), 'mean': np.array([0]), 'active': False,
         'num_par': -2},
        # {'func':New_Gauss_Blur, 'name': 'New_Gauss_Blur', 'params': np.array([3, 5, 7,9]), 'std': np.array([1.0]),'active': False, 'num_par': -1},
        # {'func':Gaussian_blur, 'name': 'Gauss_Blur', 'params': np.array([3, 5, 7,9]), 'std': np.array([1.0]),'active': False, 'num_par': -1},
        # {'func':Gaussian_Noise, 'name': 'Gauss_Noise', 'params': np.array([0,0.0141,0.02,0.0245,0.0283,0.3162,0.1]), 'mean': np.array([0]),'active': False, 'num_par': -2},
        # {'func':Salt_and_Pepper, 'name': 'salt and pepper', 'params': np.array([0.01,0.02,0.03,0.04,0.05,0.1]), 'active': False,'num_par': 1},
        {'func': Scale, 'name': "Scale", 'params': np.array([0.5,0.8,1.5,2.0]), 'active': False, 'num_par': 1},
        {'func':Rotation,'name':'rotation','params':np.array([1,2,5,10]),'active':False,'num_par':1},
        # {'func':grid_crop, 'name': 'grid_crop', 'params': np.array([0.1, 0.3, 0.5,0.7]), 'active': False, 'num_par': 1},
        # {'func':Cropout,'name':'cropout','params':np.array([0.1,0.3,0.5,0.7]),'active':False,'num_par':4},
        # {'func':Dropout,'name': 'dropout', 'params': np.array([0.1,0.3,0.5,0.7]), 'active': False, 'num_par': 2},
        # {'func':Adjust_Brightness,'name':'Adjust Brightness','params':np.array([1.1,1.2,1.3]),'active':False,'num_par':1},
        # {'func':Adjust_contrast,'name':'Adjust Contrast','params':np.array([1.0,1.5,2.0]),'active':False,'num_par':1},
        # {'func':Adjust_Saturation,'name':'Adjust_Saturation','params':np.array([1.1,1.5,2.0]),'active':False,'num_par':1},
        # {'func':Sharpening,'name':'sharpen','params':np.array([1,5,10]),'active':False,'num_par':1},
        # {'func':Scale,'name':"Scale",'params':np.array([1,1.05,1.1,1.15,1.2]),'active':False,'num_par':1}
        # {'func':Combine,''}
    ]
    key=1
    for attack in Attack_list:
        if attack['active']==False:
            continue
        for attack_params in attack['params']:
            if attack['num_par']==1:
                noiser=attack['func'](attack_params)
            elif attack['num_par']==2:
                noiser=attack['func']([attack_params,attack_params])
            elif attack['num_par']==4:
                noiser=attack['func']([np.sqrt(attack_params),np.sqrt(attack_params)],[np.sqrt(attack_params),np.sqrt(attack_params)])
            elif attack['num_par']==-1:
                noiser=attack['func'](attack_params,attack['std'])
            elif attack['num_par']==-2:
                noiser=attack['func'](attack['mean'],attack_params)
            else:
                print("Setting!")


            checkpoint = torch.load(args.checkpoint_file,map_location='cpu')
            # checkpoint = torch.load(args.checkpoint_file)
            hidden_net = Hidden(hidden_config, device, noiser, None)
            utils.model_from_checkpoint(hidden_net, checkpoint)
            source_images=os.listdir(args.source_images)

            # test_image_num=200
            error=0
            error_left=0

            psnr=0
            avg_ber=0
            avg_ncc=0
            flag=1

            test_image_num=0
            # print("{}({})".format(attack['name'], attack_params))
            # with tqdm(source_images,) as process_image:
            process_image=tqdm(source_images,leave=False)
            if flag==1:
                Cover_ZW_list=[]

            for source_image in process_image:
                process_image.set_description("{}({})".format(attack['name'],attack_params))
                # process_image.set
                if test_image_num>200:
                    break
                test_image_num=test_image_num+1
                RGB_image_path=os.path.join(args.source_images,source_image)

                image=Image.open(RGB_image_path)
                # image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)   #(256,256,3)
                transform=transforms.Compose([
                    transforms.CenterCrop((hidden_config.H,hidden_config.W)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                image_tensor=transform(image).float().to(device)    #(3,config.H,config.W)
                image_tensor.unsqueeze_(0)  #(1,3,config.H,congig.W)
                np.random.seed(42)

                losses,(cover_zw_round,noise_zw_round)=hidden_net.validate_on_batch([image_tensor,image_tensor])
                if flag==True:
                    if test_image_num==1:
                        all_cover_zw=cover_zw_round
                    else:
                        all_cover_zw=np.concatenate((all_cover_zw,cover_zw_round),axis=0)

                    # if test_image_num<8:
                    # print("B:",all_cover_zw.shape[0])
                    if all_cover_zw.shape[0]==8:

                        # test_NC, NC_max, NC_avg = F_NC(cover_zw_round)
                        test_NC, NC_max, NC_avg = F_NC(all_cover_zw)
                        test_NC_path="/home/dell/Documents/HLQ/Zerowatermark_2/Result/test_NC.xlsx"
                        result = pd.DataFrame(test_NC)
                        result.to_excel(test_NC_path, sheet_name="NC", index=False)
                        # print("nc", all_cover_zw.shape[0])
                # if test_image_num==1:
                #     all_cover_zw=cover_zw_round
                # else:
                #     all_cover_zw=np.concatenate((all_cover_zw,cover_zw_round),axis=0)
                #     print("nc",all_cover_zw.shape[0])
                # if test_image_num==10:
                #     test_NC, NC_max, NC_avg = F_NC(cover_zw_round)
                #     test_NC_path="/home/dell/Documents/HLQ/Zerowatermark_2/Result/test_NC.xlsx"
                #     result = pd.DataFrame(test_NC)
                #     result.to_excel(test_NC_path, sheet_name="NC", index=False)
                ber,ncc=my_zw_result(cover_zw_round,noise_zw_round)
                avg_ber=ber+avg_ber
                avg_ncc=ncc+avg_ncc
            avg_ber=avg_ber/test_image_num
            avg_ncc=avg_ncc/test_image_num
            attack_type=[attack['name'],attack_params]
            save_list=result_all(attack_type,avg_ber,avg_ncc,save_list)

    Result_To_Excel(save_list)



if __name__ == '__main__':
    main()

import cv2
import numpy as np
import torch.nn as nn
import torch
import os
import PIL
from PIL import *
class Smooth(nn.Module):
    def __init__(self,w_r):
        super(Smooth, self).__init__()
        self.w=w_r
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)
            noise = cv2.blur(A, (self.w, self.w))
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover
from scipy.signal import wiener
# def Wiener2(mat,w_r,w_c):
#     for k in range(batch):
#         B = img[k]
#         C = []
#         for i in range(3):
#             T = B[:, :, i]
#             T = T.astype("float64")
#             T = wiener(T, mysize=[w_r, w_c])
#             C.append(T)
#         C = np.array(C)
#         C = np.transpose(C, (1, 2, 0)).astype("int")
#         list.append(C)
#     out = np.array(list)
#     return out
class Winner(nn.Module):
    def __init__(self,w_r):
        super(Winner, self).__init__()
        self.w=w_r
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)
            temp=[]
            for c in range(3):
                T = A[:, :, c]
                T = T.astype("float64")
                T = wiener(T, mysize=[self.w, self.w])
                temp.append(T)
            noise=np.array(temp)#(3,128,128)
            # noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover
class Rotation(nn.Module):
    def __init__(self,angle):
        super(Rotation, self).__init__()
        self.angle=angle
        self.scale=1
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()

        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)

            M = cv2.getRotationMatrix2D((H // 2, W // 2), self.angle, self.scale)
            noise=cv2.warpAffine(A, M, (H, W))
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover

class New_Jpeg(nn.Module):
    def __init__(self,Q):
        super(New_Jpeg, self).__init__()
        self.Q=int(Q)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)
            # noise = cv2.blur(A, (self.w, self.w))

            # def jpg(img, Q):
            #     global exp_id
            #     cv2.imwrite('temporary_files/temp_{}.jpg'.format(exp_id), img, [cv2.IMWRITE_JPEG_QUALITY, int(Q)])
            #     Iw_attacked = cv2.imread('temporary_files/temp_{}.jpg'.format(exp_id), cv2.IMREAD_GRAYSCALE)
            #     return Iw_attacked

            # path_jpeg = "/media/dell/DOC/HLQ/New_RGBDNet_2/Result/tempor"

            # cv2.imwrite(path_jpeg + "/img_" + str(i) + ".jpg", A, [cv2.IMWRITE_JPEG_QUALITY, self.Q])
            print("Q:",self.Q)
            cv2.imwrite("Result/tempor/img_{}.jpg".format(str(i)), A, [cv2.IMWRITE_JPEG_QUALITY, self.Q])
            print("OK")
            noise = cv2.imread("Result/tempor/img_{}.jpg".format(str(i)))

            # noise = cv2.imread(path_jpeg + "/img_" + str(i) + ".jpg")
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover
#
# def Jpeg_Noise(mat,quality):
#     A=mat.copy()
#     list = []
#     batch=A.shape[0]
#     for i in range(batch):
#         C=A[i]
#         path_jpeg = "/home/dell/Documents/HLQ/Work1/images/jpeg_noise"
#         cv2.imwrite(path_jpeg + "/img" + str(i) + ".jpg", C, [cv2.IMWRITE_JPEG_QUALITY, quality])
#         B = cv2.imread(path_jpeg + "/img" + str(i) + ".jpg")
#         list.append(B)
#         os.remove(path_jpeg+"/img"+str(i)+".jpg")
#     out=np.array(list)
#     return out


class New_Gauss_Blur(nn.Module):
    def __init__(self,w_r,sigma):
        super(New_Gauss_Blur, self).__init__()
        self.w=w_r
        self.sigma=sigma
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)

            noise = cv2.GaussianBlur(A, (self.w, self.w), self.sigma)
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)

        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover


class Sharpening(nn.Module):
    def __init__(self,radius):
        super(Sharpening, self).__init__()
        self.r=radius
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)
            A=np.uint8(A)
            PIL_A = PIL.Image.fromarray(A)
            PIL_A = PIL_A.filter(PIL.ImageFilter.UnsharpMask(radius=self.r, percent=150, threshold=3))
            noise=np.array(PIL_A).astype('float32')
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)

        noise_and_cover[0]=noise_img
        return noise_and_cover



class Scale(nn.Module):
    def __init__(self,ratio):
        super(Scale, self).__init__()
        self.ratio=ratio
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()
        new_row=int(H*self.ratio)
        new_col=int(W*self.ratio)
        noise_image_list=[]
        for i in range(B):
            A = encode_image[i]
            A =A.cpu().detach().numpy().transpose(1, 2, 0)
            # M = cv2.getRotationMatrix2D((H // 2, W // 2), self.angle, self.scale)
            # noise=cv2.warpAffine(A, M, (H, W))
            out_A = cv2.resize(A, dsize=(new_col, new_row))
            noise = cv2.resize(out_A, dsize=(H, W))
            noise = noise.transpose(2,0,1) #(C,H,W)
            noise_image_list.append(noise)
        noise_img = np.array(noise_image_list) #.reshape(B,C,H,W)
        noise_img= torch.tensor(noise_img, device=self.device)
        noise_and_cover[0]=noise_img
        return noise_and_cover
from noise_layers.jpeg import *
from noise_layers.Median_filter import *
from noise_layers.salt_and_pepper import *
from noise_layers.Gaussian_noise import *
from noise_layers.jpeg_compression import JpegCompression

class Hybrid_Attack(nn.Module):
    def __init__(self,p):
        super(Hybrid_Attack, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.JP = JpegCompression(self.device)
        # self.GF=New_Gauss_Blur(7,1.0)
        # self.AF=Smooth(7)
        self.MF=Median_filter(5)
        self.SPN=Salt_and_Pepper(0.05)
        self.GN=Gaussian_Noise(0,0.05)
        self.RT=Rotation(2)
        self.SC=Scale(2.0)
    def forward(self,noise_and_cover):

        #Test1:MF(5)+SPN(0.3)
        # noise_and_cover=self.MF(noise_and_cover)
        # noise_and_cover=self.SPN(noise_and_cover)
        # #Test2:MF(5)+GN(0.3)
        # noise_and_cover = self.MF(noise_and_cover)
        # noise_and_cover = self.GN(noise_and_cover)
        # #Test3:MF(5)+JP(10)
        # noise_and_cover = self.MF(noise_and_cover)
        # noise_and_cover = self.JP(noise_and_cover)
        # #Test4:JP(10)+SPN(0.3)
        # noise_and_cover = self.JP(noise_and_cover)
        # noise_and_cover = self.SPN(noise_and_cover)
        #Test5:JP(10)+GN(0.3)
        noise_and_cover=self.JP(noise_and_cover)
        noise_and_cover=self.GN(noise_and_cover)
        # #Test6:RT(2)+JP(10)
        # noise_and_cover = self.RT(noise_and_cover)
        # noise_and_cover = self.JP(noise_and_cover)
        # #Test7:JP(10)+SC(2.0)
        # noise_and_cover = self.JP(noise_and_cover)
        # noise_and_cover =self.SC(noise_and_cover)


        return noise_and_cover

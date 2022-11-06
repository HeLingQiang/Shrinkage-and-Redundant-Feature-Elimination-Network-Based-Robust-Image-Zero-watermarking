import numpy as np
import torch.nn as nn
import torch
# def my_zw_(input_mat):
#     # in:32,1024?
#     # out¡32,1024?
#     img=input_mat.copy()
#     batch = 32
#     channel = 1
#     ave_window = 16
#     col = 256
#     for k in range(batch):
#         A = img[k]
#         A = A.reshape(ave_window, ave_window, channel)
#         for i in range(channel):
#             T = A[:, :, i]
#             aver = np.mean(T)
#             AVER = np.zeros(shape=[ave_window, ave_window]) + aver
#             E = (T > AVER).astype(int)
#             if i == 0:
#                 out = E
#             else:
#                 out = np.concatenate([out, E], axis=0)
#         out = out.reshape(1, col)
#         if k == 0:
#             OUT = out
#         else:
#             OUT = np.concatenate([OUT, out], axis=0)
#     return OUT

def my_zw(input_mat):
    img=input_mat.copy()
    batch,channel,height,width=img.shape
    block_row=height
    block_col=width
    block_num=int((height/block_row)*(width/block_col))
    for k in range(batch):
        A=img[k]
        A=np.squeeze(A)
        A=A.reshape(block_num,block_row,block_col)
        for i in range(block_num):
            avg=np.mean(A[i])
            AVER=np.zeros(shape=[block_row,block_col])+avg
            E = (A[i] > AVER).astype(int)
            E=np.expand_dims(E,axis=0)#(1,2,2)
            if i==0:
                block_zw=E
            else:
                block_zw=np.concatenate([block_zw,E],axis=0)
        A_ZW=block_zw.reshape(1,1,height,width)#(1,10,10)
        if k==0:
            out=A_ZW
        else:
            out=np.concatenate([out,A_ZW],axis=0)
    return out


def F_NC(cover_zw):
    img=cover_zw.copy()
    img=np.squeeze(img)
    batch=img.shape[0]
    NC_mat=np.zeros(shape=(batch,batch))-1
    #img(32,1,128,128)
    NC_max=0
    NC_avg=0
    for i in range(batch):
        img1=img[i]
        img1=img1+0.000000000001
        for j in range(batch):
            if i==j:
                NC_mat[i][j]=1
            if i<j:
                img2=img[j]+0.000000000001
                d1=np.sum(img1*img2)
                d2=np.sum(img1*img1)
                d3=np.sum(img2*img2)
                out=d1/(np.sqrt(d2)*np.sqrt(d3))
                NC_mat[i][j]=out
                NC_mat[j][i]=out
                if NC_max<out:
                    NC_max=out
                NC_avg=NC_avg+out
    num=batch*(batch-1)*2
    NC_avg=NC_avg/num
    return  NC_mat,NC_max,NC_avg

def F_NCC(A,B):
    img1=A.copy()+0.000000000001
    img2=B.copy()+0.000000000001
    d=np.sum(img1*img2)
    d1=np.sum(img1*img1)
    d2=np.sum(img2*img2)
    out=d/np.sqrt(d1*d2)
    return out

class Balance_loss(nn.Module):
    def __init__(self,ssim_w=0.5,mse_w=0.5,vgg_w=0):
        super().__init__()
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = nn.L1Loss().to(device)
        self.mse_w=mse_w
        self.mse_loss = nn.MSELoss().to(device)
        self.vgg_w=vgg_w
        # self.unique=loss_nc()
        # self.vgg_loss=VGGLoss_2()
    def forward(self,cover_zw,noise_zw):
        mse=self.mse_loss(cover_zw,noise_zw)
        g_loss=mse
        # mean=loss_nc(cover_zw)
        # mean=np.mean(cover_zw)
        # print("mean",mean)
        return g_loss


# import numpy as np
# import cupy as cp
import utils
import torch
from model.formal_def import *
from utils import *
import pandas as pd
from tqdm import *
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch
import matplotlib
# matplotlib.use('Agg')
import numpy as np

def result_all(attack_type, avg_ber,avg_ncc, save_list: list):

    print("{}({}): [{:.5},{:.5}]".format(attack_type[0],attack_type[1],avg_ber,avg_ncc))
    save_data={'Name':attack_type[0]+"("+str(attack_type[1])+")","BER":avg_ber,"NCC":avg_ncc}
    save_list.append(save_data)
    return save_list

def Result_To_Excel(save_list:list):
    save_path="/home/dell/Documents/HLQ/Zerowatermark_2/Result"+"/BER_NCC.xlsx"
    Name=[]
    BER_list=[]
    NCC_list=[]
    process_save_list=tqdm(save_list,leave=False)
    for data in process_save_list:
        process_save_list.set_description("{}".format(data['Name']))
        attack_name=data['Name']
        ber=data['BER']
        ncc=data['NCC']
        Name.append(attack_name)
        BER_list.append(ber)
        NCC_list.append(ncc)
    merge=[Name,BER_list,NCC_list]
    result = pd.DataFrame(merge).T
    result.to_excel(save_path, sheet_name="BER_NCC",index=False)
    print("Success Save!")

def my_zw_result(cover_zw,noise_zw):

    cover_zw_round=cover_zw
    noise_zw_round=noise_zw

    batch_size=cover_zw_round.shape[0]
    row = cover_zw_round.shape[2]
    col = cover_zw_round.shape[3]
    BER = np.sum(np.abs(cover_zw_round - noise_zw_round)) / (batch_size * row * col)
    NCC = F_NCC(cover_zw_round, noise_zw_round)

    return BER ,NCC

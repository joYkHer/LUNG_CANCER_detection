import numpy as np

all_p_slices = np.load('./pre_pslices.npy', allow_pickle=True)

def normalize_one_patient_slices(pslices):
    '''
    将一个patient的slices (b-mean)/std， maybe? just have a try...; 缩放到-1~1范围内
    数据划分为两部分, tissue和air, 让air保持为0, 只对tissue进行缩放
    :param pslices: 一个病人的所有slices, 三维
    :return: 标准化后的slices, numpy类型
    '''
    pslices[pslices != 0] = (pslices[pslices != 0] - np.mean(pslices[pslices != 0])) / np.std(pslices[pslices != 0])
    # 对tissue部分(x-mean)/std
    tissue_min = np.min(pslices[pslices!=0])
    tissue_max = np.max(pslices[pslices!=0])
    a = 2/(tissue_max - tissue_min); b = (-tissue_min-tissue_max)/(tissue_max - tissue_min)
    pslices[pslices!=0] = pslices[pslices!=0]*a+b
    # 对tissue部分进行-1~1的缩放, 缩放方法: y = a*x+b (min,-1),(max,1)
    return pslices

def normalize_all_patients(all_pslices):
    return np.asarray([
        normalize_one_patient_slices(pslices)
        for pslices in all_pslices
    ])

def save_normalized_psilces(fName):
    np.save(fName,normalize_all_patients(all_p_slices), allow_pickle=True)

if __name__=='__main__':
    save_normalized_psilces("pre_normalized_pslices.npy")
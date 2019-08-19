import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

all_normalized_pslices = np.load(r'./pre_normalized_pslices.npy', allow_pickle=True)
pid_to_label = np.load(r'./pid_to_label_one.npy', allow_pickle=True).tolist()
# 到时候还是按照label的顺序来读取好了，免得各种原因顺序乱了

class custom_dataset(Dataset):
    def __init__(self, all_p_slices, pid_to_labels, transform): # transform, 一个实例
        super(custom_dataset, self).__init__()
        self.all_patient_slices = all_p_slices
        self.pid_to_lable = pid_to_labels
        self.transform = transform

    def __len__(self): # len(class_instance)
        return len(self.all_patient_slices) # part:205个

    def __getitem__(self, idx): # class_instance[i]
        '''
        __getitem__ 返回第i个样本么, 那默认的参数item是下标了.
        :param idx: self.all_patient_slices的下标
        :return: 病人和对应的label
        '''
        assert os.listdir(r'./stage1') == list(pid_to_label.keys()), 'ooops'
        # 保证与最开始的目录顺序一致， 不要手贱去改..
        # self.transform 得定义__call__()...
        return self.transform({'slices':self.all_patient_slices[idx], 'label': list(pid_to_label.values())[idx]}) # transform实例__call__()


class to_tensor(object):
    def __init__(self):
        super(to_tensor, self).__init__()

    def __call__(self, pslices_and_label): # 定义__call__,  实例()就是嗲用该函数
        # type()为字典, pslices: array, label:str
        pslices, label = pslices_and_label['slices'], pslices_and_label['label']
        #pslices = pslices.transpose((2,0,1))
        # numpy image: H x W x C
        # torch image: C X H X W
        # TODO need to check out, codes below too
        return {'slices':torch.from_numpy(pslices), 'label': torch.tensor(int(label))}


p_dataset = custom_dataset(
    all_normalized_pslices,
    pid_to_label,
    transform=transforms.Compose([
        to_tensor()
    ]))
print(p_dataset[0])
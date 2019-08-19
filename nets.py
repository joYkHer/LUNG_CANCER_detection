import numpy as np
from  torch.utils.data import Dataset
from torchvision import transforms
import os

all_normalized_pslices = np.load(r'./pre_normalized_pslices.npy', allow_pickle=True)
pid_to_label = np.load(r'./pid_to_label_one.npy', allow_pickle=True).tolist()
# 到时候还是按照label的顺序来读取好了，免得各种原因顺序乱了

class custom_dataset(Dataset):
    def __init__(self, all_p_slices, pid_t0_labels):
        super(custom_dataset, self).__init__()
        self.all_patient_slices = all_p_slices
        self.pid_to_lable = pid_t0_labels

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
        return { 'slices':self.all_patient_slices[idx], 'label': list(pid_to_label.values())[idx]}


p_dataset = custom_dataset(all_normalized_pslices, pid_to_label)
print(p_dataset[0])
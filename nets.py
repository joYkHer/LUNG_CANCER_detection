import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # Dataset定义数据集,对数据的操作, DataLoader定义怎么拿数据
from torchvision import transforms
from torch.autograd import Variable
import os

all_normalized_pslices = np.load(r'./pre_normalized_pslices.npy', allow_pickle=True)
pid_to_label = np.load(r'./pid_to_label_one.npy', allow_pickle=True).tolist()
BATCH_size = 4
NUM_WORKS = 2
new_pic_size = 128
scaled_z_hat = 20 # 这两个整个项目保持同步
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
        pslices = np.reshape(pslices, (1,scaled_z_hat,new_pic_size,new_pic_size)) # 灰度图, Channel为1
        return {'slices':torch.from_numpy(pslices).type(torch.DoubleTensor), 'label': torch.tensor(int(label))}


p_dataset = custom_dataset(
    all_normalized_pslices,
    pid_to_label,
    transform=transforms.Compose([
        to_tensor()
    ])) # dataset.transform -> trainsform.Compose([func...])->func.__call__() // 注意下funcs之间参数的传递

p_dataloader = DataLoader(
    p_dataset,
    batch_size=BATCH_size,
    shuffle=False, # label的分布已经random的了? TODO need to check out
 #   num_workers=NUM_WORKS
)

# p_slices_batch:torch.Size([4, 1, 20, 128, 128])
class BaseLineNet(nn.Module):
    def __init__(self):
        super(BaseLineNet, self).__init__()
        # self.classconv = nn.Sequential(
        #     nn.Conv3d(1, 16, 4, stride=2),
        #     # nn.ReLU(), # relu? tissue变air?不太好吧..
        #     nn.MaxPool3d(2, 2),
        #     nn.Conv3d(16, 64, 4, stride=2),
        #     nn.MaxPool3d(2, 2),
        #     nn.Conv3d(64, 128, 4, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2, 2)
        # )

        self.conv1 = nn.Conv3d(1, 16, (1,2,2), stride=2)
        # nn.ReLU(), # relu? tissue变air?不太好吧..
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 64, (1,2,2), stride=2)
        #nn.MaxPool3d(2, 2),
        self.conv3 = nn.Conv3d(64, 128, (1,2,2), stride=2)
        #nn.ReLU(),
        #nn.MaxPool3d(2, 2)

        # self.fullconnect = nn.Sequential(
        #     nn.Linear()
        # )
    def weight_init(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv3d):
                mod.weight = nn.Parameter(mod.weight.double())
                mod.bias = nn.Parameter(mod.bias.double())

    def forward(self, pslices):
        self.weight_init()
        x = self.conv1(pslices)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

# torch.Size([4, 128, 1, 2, 2])

baseline = BaseLineNet()
for p_slices_label_batch in p_dataloader:
    print(baseline(p_slices_label_batch['slices']).shape)
    break
    # print(p_slices_label_batch['slices'].type(torch.DoubleTensor))
    # break
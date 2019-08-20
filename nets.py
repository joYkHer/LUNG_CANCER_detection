import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Dataset定义数据集,对数据的操作, DataLoader定义怎么拿数据
from torchvision import transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_normalized_pslices = np.load(r'./pre_normalized_pslices.npy', allow_pickle=True)
pid_to_label = np.load(r'./pid_to_label_one.npy', allow_pickle=True).tolist() # 字典保存为npy文件, np.load()的时候只要tolist()就从array变回字典了,很神奇
BATCH_SIZE = 5 # 205 能整除5
NUM_WORKS = 2
new_pic_size = 128
scaled_z_hat = 20 # 这两个整个项目保持同步
LEARNING_RATE = 1e-3
EPOCH = 10 # TODO headache
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
        return {'slices':torch.from_numpy(pslices).type(torch.DoubleTensor), 'label': torch.tensor(int(label)).double()}


p_dataset = custom_dataset(
    all_normalized_pslices,
    pid_to_label,
    transform=transforms.Compose([
        to_tensor()
    ])) # dataset.transform -> trainsform.Compose([func...])->func.__call__() // 注意下funcs之间参数的传递

p_dataloader = DataLoader(
    p_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # label的分布已经random的了? TODO need to check out
 #   num_workers=NUM_WORKS
)

# p_slices_batch:torch.Size([4, 1, 20, 128, 128])
class BaseLineNet(nn.Module):
    def __init__(self):
        super(BaseLineNet, self).__init__()
        self.classconv = nn.Sequential(
            nn.Conv3d(1, 16, (1,2,2), stride=2),
            # nn.ReLU(), # relu? tissue变air?不太好吧..
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 64, (1,2,2), stride=2),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, (1,2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )
        # torch.Size([4, 128, 1, 2, 2])
        self.fullconnect = nn.Sequential(
            nn.Linear(128*1*2*2,128),
            nn.Dropout(0.5),
            nn.Linear(128,32),
            nn.Linear(32,1)
        )

    # def float_to_double(self, mod_and_mod_to_handle): # float64 就是double
    #     mod, mod_to_handle = mod_and_mod_to_handle
    #     if isinstance(mod, mod_to_handle):
    #         mod.weight, mod.bias = nn.Parameter(mod.weight.double()), nn.Parameter(mod.bias.double())
    #
    # def modules_float_to_double(self, modules, mod_to_handle):
    #     # 这里也不需要map的返回值, 只要操作
    #     map(self.float_to_double,zip(modules, [mod_to_handle]*len(modules)))
    #
    # def weight_init(self):
    #     self.modules_float_to_double(self.classconv, nn.Conv3d)
    #     self.modules_float_to_double(self.fullconnect, nn.Linear)
    # map function is definitly a sad story, it seems change nothing, only variables the functions return care.
    # map函数看起来不改变任何原有变量，要获得函数处理的数值，就要定义在map里面的fun函数return的变量, var = map(...),list(map(...))也是， ps zip()返回的其实是list

    def float_to_double(self, module, type_to_handle):
        if isinstance(module, type_to_handle):
            module.weight, module.bias = nn.Parameter(module.weight.double()), nn.Parameter(module.bias.double())

    def modules_float_to_double(self, modules, mod_to_handle):
        for mod in modules: self.float_to_double(mod, mod_to_handle)

    def weight_init(self):
        self.modules_float_to_double(self.classconv, nn.Conv3d)
        self.modules_float_to_double(self.fullconnect, nn.Linear)

    def forward(self, pslices):
        self.weight_init()
        x = self.classconv(pslices)
        x = x.view(BATCH_SIZE, -1)
        x = self.fullconnect(x)
        return torch.sigmoid(x)


baseline = BaseLineNet().to(device)
optimizer = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)

def train_baseline(epoch):
    baseline.train()
    for batch_id, pslices_and_label_batch in enumerate(p_dataloader):
        pslices_batch, label_batch = pslices_and_label_batch['slices'].to(device), pslices_and_label_batch['label'].to(device)
        optimizer.zero_grad()
        batch_output = baseline(pslices_batch).view(BATCH_SIZE)
        loss = F.binary_cross_entropy(batch_output, label_batch) # TODO checkout loss += 每个batch,loss都重新计算?
        # F.binary_cross_entropy() 对应二分类, F.cross_entropy() 多分类?
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print('Train Epoch:{0}, batch_id:{1}, loss:{2}'.format(epoch, batch_id, loss))

if __name__=="__main__":
    for i in range(EPOCH):
        train_baseline(i)

# for p_slices_label_batch in p_dataloader:
#     print(baseline(p_slices_label_batch['slices']))
#     break
    # print(p_slices_label_batch['slices'].type(torch.DoubleTensor))
    # break
import os, pydicom, cv2
import numpy as np # numpy.reshape 和 cv2.resize差别，numpy：shape相乘不变， cv2 可变，从图上看就是图片缩小了

pid_to_label = np.load('pid_to_label_one.npy')
new_pic_size = 128 # 最好2^n
scaled_z_hat = 20 # changeable, of course
'''
{ 
    '0015ceb851d7251b8f399e39779d1e7d': 1,
    '0030a160d58723ff36d73f41b170ec21': 0,
    ...
}
'''

def one_patient_producer(path = r'./stage1'):
    '''
    产生一个patient的pid, 默认是第一个patient
    其实可一个random一个出来...改[0]
    :param path: 所有patient的id文件路径
    :return: 一个patient的pid
    '''
    return os.listdir(path)[0]


def load_slices(pid, path = r'./stage1'):
    '''
    一次读取一个病人的所有slices, 将这些slices按照Z轴的顺序排好序
    :param path: 要读取的文件路径
    :param pid:  要读取哪一个病人的所有切片
    :return: 排好序之后的slices
    '''
    ppath = os.path.join(path, pid)
    pslices = [ pydicom.read_file(os.path.join(ppath, dcm)) for dcm in os.listdir(ppath)]
    pslices.sort(key= lambda x: int(x.ImagePositionPatient[2]))
    return pslices # 得到的type都是<class 'pydicom.dataset.FileDataset'>

def dcm_attrs(dcmpath = r'D:\works\stage1\00cba091fa4ad62cc3200a657aeb957e\0a291d1b12b86213d813e3796f14b329.dcm'):
    '''
    显示一个dcm文件的所有meta, 或者属性
    :param dcm: 硬编码的一个dcm文件路径
    :return: 指定dcm文件的attr
    '''
    pdcm = pydicom.read_file(dcmpath)
    return dir(pdcm)

def resize_one_patient_slices(new_size, pslices):
    '''
    将一个patient的所有slices调整为新的size, 像素太大运算不过来吧
    :param new_size: 重新选择的正方形size
    :param pslices: 指定patient的所有slices
    :return: pid病人所有的resize_slices
    '''
    pslices_resize = [ cv2.resize(slice.pixel_array,(new_size, new_size))  for slice in pslices] # slice都是dcm文件
    return pslices_resize  # resize之后全部都是<class 'numpy.ndarray'>...cv2.resize有点神奇，整体是list

def unk_to_air(pslices, unk = -2000, air = 0):
    '''
    数据内部将非air非tissue的地方设置为-2000,将其改为0，相当于胸腔外的为air，要找胸腔内奇怪的tissue
    :param unk:scanner默认的unk数值
    :param air:空气数值
    :return: 将unk转成air后的slices
    '''
    pslices = np.asarray(pslices)
    pslices[pslices==unk] = air # 转成numpy才能使用此方法
    return pslices # 这下所有的都是numpy了


class z_hat_scaler(object):
    def __init__(self, pslices, scaled_z_hat):
        super(z_hat_scaler, self).__init__()
        self.pslices = pslices # self.pslices得是全是arrange
        self.scaled_z_hat = scaled_z_hat
        self.pslices_scared = []

    def yield_n_slices(self, n):
        for i in range(0, len(self.pslices), n):
            yield self.pslices[i:i+n] # 最后一次不满n的情况

    def mean(self, np_list):
        '''
        注意np_list的结构
        :param np_list: [arrange, arrange, arrange..... ]
        :return:  arrange+arrange..... / np_list元素个数，相当于Z轴对应取平均
        '''
        return sum(np_list) / len(np_list)

    def zero_slice_pooling(self): # 用air/0进行填充
        if len(self.pslices) % self.scaled_z_hat != 0:  # 无法整除，对slices进行填充
            z_need_to_pool = self.scaled_z_hat - (len(self.pslices) % self.scaled_z_hat)
            slices_to_pool = np.zeros((z_need_to_pool, self.pslices.shape[1], self.pslices.shape[2]))
            self.pslices = np.concatenate((self.pslices,slices_to_pool))

    def scale_z_hat(self):
        '''
        每个patient的slices个数都不一样，将他们统一到规定数量的slice内，有点像是丢失精度
        :param scaled_z:  指定的slice个数
        :return: scale之后的slices
        '''
        self.zero_slice_pooling() # 先进行0填充
        for slices_part in self.yield_n_slices(int(len(self.pslices) / self.scaled_z_hat)): #  这里改成//也行吧
            slices_part_scared = list(map(self.mean, zip(*slices_part)))
            self.pslices_scared.append(slices_part_scared)

    def scale_run(self):
        self.scale_z_hat()
        self.pslices_scared = np.asarray(self.pslices_scared)

if __name__ == '__main__': # dcm文件可能有shape的metadata
    pslices = resize_one_patient_slices(new_pic_size,load_slices(pid= one_patient_producer()))
    #pslices = load_slices(pid=one_patient_producer())
    pslices = unk_to_air(pslices)
    pslices_scaler = z_hat_scaler(pslices, scaled_z_hat)
    pslices_scaler.scale_run()
    print(pslices_scaler.pslices_scared)
    print(pslices_scaler.pslices_scared.shape)

    # one_patient_producer() -> load_slices() -> resize_one_patient_slices() -> unk_to_air() -> z_hat_scaler.scale_run()
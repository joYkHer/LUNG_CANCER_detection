import os, pydicom, cv2
import numpy as np # numpy.reshape 和 cv2.resize差别，numpy：shape相乘不变， cv2 可变，从图上看就是图片缩小了

pid_to_label = np.load('pid_to_label_one.npy')
new_pic_size = 128 # 最好2^n
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
    return pslices_resize  # resize之后全部都是<class 'numpy.ndarray'>...cv2.resize有点神奇



if __name__ == '__main__': # dcm文件可能有shape的metadata
    #pslices = resize_one_patient_slices(new_pic_size,load_slices(pid= one_patient_producer()))
    pslices = load_slices(pid=one_patient_producer())
    print(type(pslices[0]))
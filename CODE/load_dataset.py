import torch
import torch.utils.data as Data
from torch.autograd import Variable
import random
import numpy as np
import os


def generate_filenames(mode, start_i):
    filenames = []
    if mode == 'test':
        for i in range(start_i, 26000, 1000):
            filename = 'vg_{}_{}-{}.txt'.format(mode, i, i+1000)
            filenames.append(filename)
        filenames.append('vg_test_26000-26446.txt')
    elif mode == 'train':
        for i in range(start_i, 57000, 1000):
            filename = 'vg_{}_{}-{}.txt'.format(mode, i, i+1000)
            filenames.append(filename)
        filenames.append('vg_train_57000-57723.txt')
        np.random.shuffle(filenames)
    return filenames

class MyDataset(Data.Dataset):
    def __init__(self, mode, filenames, num_classes):
        self.file_path = '{}/{}_sgcls_txt'.format(mode, mode)
        self.filenames = filenames # a list of filenames
        self.file_id = 0
        self.feat = None
        self.feat_id = 0
        self.img_id = 0
        self.mode = mode
        self.num_classes = num_classes

    def load_file(self):
        if self.file_id >= len(self.filenames):
            return
        feat_f = open(os.path.join(self.file_path, self.filenames[self.file_id]), 'r')
        self.feat = feat_f.readlines()
        self.file_id += 1
        self.feat_id = 0

    def __len__(self):
        return 670591 if self.mode == 'train' else 325570

    def __getitem__(self, item):
        if self.feat is None or self.feat_id >= len(self.feat):
            self.load_file()
        data = self.feat[self.feat_id].strip().split(' ')
        data = np.asarray(data, dtype = np.float)
        self.feat_id += 1
        box_fea = data[7:-2]
        box_fea[-4] /= data[-1]  # x1/w
        box_fea[-3] /= data[-2]  # y1/h
        box_fea[-2] /= data[-1]  # x2/w
        box_fea[-1] /= data[-2]  # y2/h
        box_label = np.zeros(self.num_classes)
        box_label[int(data[2])] = 1
        return np.array([data[0]]), torch.from_numpy(box_fea), torch.from_numpy(box_label)


if __name__=="__main__":
    mode = 'train'
    filenames = generate_filenames(mode)

    epochs = 1
    batch_size = 1

    for epoch in range(epochs):
        train_dataset = MyDataset(mode, filenames)
        train_iter = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
        cnt_binary_pos = 0
        cnt_box = 0
        for i, (_, x, y) in enumerate(train_iter):
            x = x[0].numpy()    # feat x.shape=(64,4259)
            y = y[0].numpy()    # edge
            # print(epoch, i, x.shape, y.shape)
            label = x[:, 2]
            cnt_box += len(label)
            cnt_binary_pos += sum(label)
            xt = torch.LongTensor(np.array(label).reshape(1,-1))[0]
            print(sum(label), sum(xt))
            if i == 999:
                print(cnt_binary_pos, cnt_box)
                break
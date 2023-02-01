import json
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms

from function import read_ann_json

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X,image_id,std_mass,section,offset in train_loader:
        X = X.float()
        for d in range(3):
            max = X.max()
            X = X.div(max)
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


class ECUSTFD(data.Dataset):
    def __init__(self, dir, anns, phase,category):
        self.dir = dir
        self.anns = anns
        # phase: train, val, test
        assert phase in ('train',  'test')
        self.phase = phase
        assert category in ("weight(g)","volume(mm^3)")
        self.category = category

    def __getitem__(self, item):
        pretrained_size = [256, 256]
        pretrained_means = [0.485, 0.499, 0.431]
        pretrained_stds = [0.229, 0.224, 0.225]
        img_transforms = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            # must be !
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])
        if self.phase == 'train':
            data = self.anns[item]
            image_id = data['id']
            img_type = data['type']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, image_id+".JPG")
            img_a = Image.open(img_path)
            img_a = img_transforms(img_a)
        else:
            data = self.anns[item]
            image_id = data['id']
            img_type = data['type']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, image_id+ ".JPG")
            img_a = Image.open(img_path)
            img_a = test_transforms(img_a)
            return img_a,image_id,std_mass,section,offset
        return img_a,image_id,std_mass,section,offset

    def __len__(self):
        return len(self.anns)



if __name__ == '__main__':
    # train_ls,train_picture_number,train_anchor = get_anchor(10,'train','calories')
    # test_ls, test_picture_number, test_anchor = get_anchor(10, 'test','calories')
    # print(train_anchor)
    # print(test_anchor)

    train_img_dir = '/disk/btc010001/ECUSTFD/ECUSTFD-resized--master/TrainImage'
    test_img_dir = '/disk/btc010001/ECUSTFD/ECUSTFD-resized--master/TestImage'

    train_json_file = '/disk/btc010001/ECUSTFD/data/ECUSTFD_'+'weight(g)'+'_train.json'
    test_json_file = '/disk/btc010001/ECUSTFD/data/ECUSTFD_'+'weight(g)'+'_test.json'

    train_anchor = [(26.0, 39.8, 32.9), (39.9, 54.5, 47.2), (54.7, 65.6, 60.15), (66.2, 94.2, 80.2), (96.2, 150.5, 123.35), (150.5, 155.1, 152.8), (156.9, 196.0, 176.45), (197.5, 214.5, 206.0), (218.0, 238.0, 228.0), (238.0, 448.0, 343.0)]
    test_anchor = [(26.0, 39.6, 32.8), (39.8, 52.4, 46.099999999999994), (54.5, 70.9, 62.7), (78.3, 92.3, 85.3), (93.2, 104.3, 98.75), (105.8, 154.2, 130.0), (154.2, 177.3, 165.75), (179.3, 219.5, 199.4), (220.5, 255.0, 237.75), (255.0, 448.0, 351.5)]

    #### dataset #####
    train_anns = read_ann_json(train_json_file)
    train_fun = ECUSTFD(train_img_dir, train_anns, phase='train',category='weight(g)')

    test_anns = read_ann_json(test_json_file)
    test_fun = ECUSTFD(test_img_dir, test_anns, phase='test',category='weight(g)')
    #### dataset #####


    #train_dataset = ImageFolder(root=r'./data/food/', transform=None)
    print(getStat(test_fun))

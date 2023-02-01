import json
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
import math

def read_ann_json(file):
    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']
    return annotations

class VFD(data.Dataset):
    def __init__(self, dir, anns, phase,category):
        self.dir = dir
        self.anns = anns
        # phase: train, val, test
        assert phase in ('train',  'test')
        self.phase = phase
        assert category in "volume"
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
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, str(image_id)+".jpg")
            img_a = Image.open(img_path)
            img_a = img_transforms(img_a)
        else:
            data = self.anns[item]
            image_id = data['id']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, str(image_id)+ ".jpg")
            img_a = Image.open(img_path)
            img_a = test_transforms(img_a)
            return img_a,image_id,std_mass,section,offset
        return img_a,image_id,std_mass,section,offset

    def __len__(self):
        return len(self.anns)


def get_anchor(segment,t,phase):
    if phase == 'train':
        with open('./dataset/vfdl_Bin_number10/VFD_volume_train_seg10.json', 'rb') as f:
            json_data = json.load(f)
            ann = json_data['annotations']
            ls = list(ann)
    else:
        if phase == 'test':
            with open('./dataset/vfdl_Bin_number10/VFD_volume_test_seg10.json', 'rb') as f:
                json_data = json.load(f)
                ann = json_data['annotations']
                ls = list(ann)

    train_picture_size = math.ceil(len(ls)/(segment))
    new_list = ls
    idx = 0
    train_list_anchor = []
    while idx<len(new_list):
        min_ = new_list[idx][t]
        idx = idx + train_picture_size-1
        if len(new_list)-idx>=train_picture_size:
            max_ = new_list[idx][t]
        else:
            max_ = new_list[-1][t]
            idx = len(new_list)
        anchor_set = (min_+max_)/2
        train_list_anchor.append((min_,max_,anchor_set))
    return train_list_anchor



if __name__ == '__main__':
    # new_list,train_picture_number,train_anchor = get_anchor(10,'train','calories')
    # test_ls, test_picture_number, test_anchor = get_anchor(10, 'test','calories')
    # print(train_anchor)
    # print(test_anchor)
    train_img_dir = ''
    test_img_dir = ''

    train_json_file = ''
    test_json_file = ''

    #### dataset #####
    train_anns = read_ann_json(train_json_file)
    train_fun = VFD(train_img_dir, train_anns, phase='train',category='volume')

    test_anns = read_ann_json(test_json_file)
    test_fun = VFD(test_img_dir, test_anns, phase='test',category='volume')
    #### dataset #####


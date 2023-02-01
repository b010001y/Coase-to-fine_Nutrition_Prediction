'''
Used to import data in .json format
'''

import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
from os import path
import json

class nutrition5k(data.Dataset):
    def __init__(self, dir, anns, phase,category):
        self.dir = dir
        self.anns = anns
        # phase: train, val, test
        assert phase in ('train',  'test')
        self.phase = phase
        assert category in ("mass","calories","fat","carb","protein")
        self.category =category

    def __getitem__(self, item):
        pretrained_size = [256, 256]
        pretrained_means = [0.467, 0.450, 0.418]
        pretrained_stds = [0.229, 0.240, 0.260]
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
        # todo: check
        if self.phase == 'train':
            data = self.anns[item]
            image_id = data['dish_id']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img = Image.open(os.path.join(self.dir, image_id,"rgb.png"))
            img = img_transforms(img)
        else:
            data = self.anns[item]
            image_id = data['dish_id']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img = Image.open(os.path.join(self.dir, image_id,"rgb.png"))
            img = test_transforms(img)

            return img,image_id,std_mass,section,offset
        return img,image_id,std_mass,section,offset

    def __len__(self):
        return len(self.anns)

####There this function not be used #####
def ReadCsvData(filepath):
    if not path.exists(filepath):
        raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 2):
            column = [row[i] for row in file]
            parsed_data[column[0]] = column
            f_in.seek(0)
    return parsed_data
####There this function not be used #####

if __name__ == '__main__':
    import json
    import torch
    import csv
    dir = '../../data/imagery/new_realsense_overhead'
    file = 'nutrition_Bin_number10/change_mass_train10.json'

    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']

    train_fun = nutrition5k(dir, annotations, phase='train',category='mass')
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size = 4, shuffle=True)

    for img,image_id,std_mass,section,offset in train_loader:
        img = img.cuda()
        print(type(img), img.shape)
        print(image_id,std_mass,section,offset)


import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import namedtuple
import numpy as np
import json
import random
from torch.backends import cudnn
import shutil
import time

from config.args_parse import args_parser
from core.function import train,read_ann_json,evaluate,MylossFunc
from models.network import ResNet, Bottleneck
from dataset.dataset import nutrition5k
from criterion.compute_eval_statistics import calMAE
from utils.util import get_anchor

CATEGORY = ["mass","calories","fat","carb","protein"]
#CATEGORY = ["fat","carb","protein"]

### set certain seed ###
def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn related setting
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True

def main(args,i,ticks):
    fix_randomness(args.seed)
    torch.cuda.empty_cache()
    batch_size = args.batch_size
    output_dim = args.output_dim
    found_lr = args.found_lr
    segment = args.segment
    epoch = args.epoch

    img_dir = './../data/imagery/new_realsense_overhead'

    train_json_file = r'./dataset/nutrition_Bin_number' + str(segment) + '/change_' + i + '_train' + str(segment)+'.json'
    test_json_file = r'./dataset/nutrition_Bin_number' + str(segment) + '/change_' + i + '_test' + str(segment)+'.json'
    train_anchor = get_anchor(segment, 'train', i)
    test_anchor = get_anchor(segment, 'test', i)

    #### dataset #####
    train_anns = read_ann_json(train_json_file)
    train_fun = nutrition5k(img_dir, train_anns, phase='train',category=i)
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size=batch_size, shuffle=True,num_workers = 8,pin_memory = True)

    test_anns = read_ann_json(test_json_file)
    test_fun = nutrition5k(img_dir, test_anns, phase='test',category=i)
    test_loader = torch.utils.data.DataLoader(test_fun, batch_size=batch_size, shuffle=False,num_workers = 8,pin_memory = True)
    #### dataset #####


    ##### network #####
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    resnet50 = models.resnet50(pretrained=True)
    model = ResNet(resnet50_config, segment)
    # Read the parameter
    pretrained_dict = resnet50.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    # Load the state_dict we really need
    model.load_state_dict(model_dict,True)

    model = model.cuda()
    learnable_pam = MylossFunc()
    #### network #####


    #### optimizer #####
    params = [
        {'params': model.conv1.parameters(), 'lr': found_lr / 10},
        {'params': model.bn1.parameters(), 'lr': found_lr / 10},
        {'params': model.layer1.parameters(), 'lr': found_lr / 8},
        {'params': model.layer2.parameters(), 'lr': found_lr / 6},
        {'params': model.layer3.parameters(), 'lr': found_lr / 4},
        {'params': model.layer4.parameters(), 'lr': found_lr / 2},
        {'params': model.fc0.parameters(), 'lr': found_lr / 2},
        {'params': model.fc1.parameters(), 'lr': found_lr / 2},
        {'params': model.fc2.parameters(), 'lr': found_lr / 2},
        {'params': model.lastlayer.parameters()},
        {'params': learnable_pam.parameters()}
    ]

    #optimizer = optim.RMSprop(params, lr=found_lr,momentum= 0.9,weight_decay= 0.9,eps=1.0)
    #optimizer = optim.Adam(params, lr=found_lr,weight_decay=0.9,eps=1.0)
    optimizer = optim.Adam(params, lr=found_lr)

    #### optimizer #####

    food_MAE = 1000
    food_MAE_pre = 100
    for iepoch in range(epoch):
        loss_sec,loss_off = train(args,train_loader, model,learnable_pam, optimizer,batch_size,segment,train_anchor)
        print('epoch ' + str(iepoch) + '\n'+ f'\tTrain Section_Loss: {loss_sec:.3f}'+'\n'+f'\tTrain Offset_Loss: {loss_off:.3f}\n')
        pre_path = evaluate(test_loader,model,learnable_pam,batch_size,segment,test_anchor,i,ticks)

        tmp_MAE,tmp_MAE_pre = calMAE(pre_path,i)
        with open("file/" + ticks + "Baseline_prediction.txt", "a") as f:
            f.write(f'{tmp_MAE:.3f}   {tmp_MAE_pre:.3f}%\n')
        if(food_MAE_pre>tmp_MAE_pre):
            food_MAE = tmp_MAE
            food_MAE_pre = tmp_MAE_pre
            print(f'\t '+i+f'_MAE:{food_MAE:.3f}\n' + f'\t '+i+f'_MAE:{food_MAE_pre:.3f}')
            #state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':iepoch}
            #torch.save(model,i+'model.pth')
    print(f'\t ' + i + f'_MAE:{food_MAE:.3f}\n' + f'\t ' + i + f'_MAE:{food_MAE_pre:.3f}')

if __name__ == '__main__':
    args= args_parser()
    segment = args.segment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    print(ticks)
    with open("file/" + ticks + "Baseline_prediction.txt","a") as f:
        f.write("data   percent\n")

    for i in CATEGORY:
        main(args,i,ticks)

    tick2 = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    print(tick2)

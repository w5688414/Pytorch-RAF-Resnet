from __future__ import print_function
from re import L

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torchvision import datasets, models
import numpy as np
import os
import pandas as pd
import torch.utils.data as data
import argparse, random
import utils
from utils import AverageMeter, RandomFiveCrop
import cv2
from RAF import RafDataSet
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
raf_path = 'datasets/raf-basic'
parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--datapath', type=str, default=raf_path, help='CNN architecture')
parser.add_argument('--data', type=str, default='SAVE', help='CNN architecture')
parser.add_argument('--model', type=str, default='resnet18', help='CNN architecture')
parser.add_argument('--bs', default=64, type=int, help='batchsize')
parser.add_argument('--cent', default=0.01, type=float, help='beta')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--savefile', type=str, default='RAF_mean_l22c7_l3c7_r196h7_l4c7_h7.t7', help='CNN architecture')
parser.add_argument('--resumefile', type=str, default='RAF_base.pth', help='CNN architecture')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument("--local_rank", type=int, default=0)
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_val_acc = 0  # best val accuracy
best_val_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
Best_value = 0 # best val accuracy

if opt.resume:
    learning_rate_decay_start= 5  # 50
else:
    learning_rate_decay_start= 10  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

#cut_size = 140
total_epoch = 300

path = os.path.join('SAVE_resnet', opt.data + '_' + opt.model)

# Data
print('==> Preparing data..')

data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        RandomFiveCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
data_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
traindir = RafDataSet(opt.datapath, phase = 'train', transform = data_transforms, trans=data_transforms, basic_aug = True)

trainloader = torch.utils.data.DataLoader(traindir,
                                        #    sampler=train_sampler,
                                           batch_size = opt.bs,
                                           num_workers = opt.workers,
                                           shuffle = True,
                                           pin_memory = False)

data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
data_trans_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
valdir = RafDataSet(opt.datapath, phase = 'test', transform = data_transforms_val, trans = data_transforms_val)

valloader = torch.utils.data.DataLoader(valdir,
                                               batch_size = opt.bs,
                                               num_workers = opt.workers,
                                               shuffle = False,
                                               pin_memory = False)


def model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 8)

    return model_ft

# Model
net = model()
net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,opt.resumefile))
    print(checkpoint.keys())


    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
           if torch.is_tensor(v):
              state[k] = v.cuda()
    best_val_acc = checkpoint['best_RAFval_acc']
    print("best_RAFval_acc= %0.3f" % best_val_acc)
    best_val_acc_epoch = checkpoint['best_RAFval_acc_epoch']

    total_epoch = start_epoch + 50
    start_epoch = 0
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets,indexes) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        # compute output
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    Train_acc = 100.*float(correct)/float(total)

def val(epoch):
    global val_acc
    global best_val_acc
    global best_val_acc_epoch
    global Best_value
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets,indexes) in enumerate(valloader):
        bs, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = net(inputs)

        loss = criterion(outputs, targets)


        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    val_acc = 100.*float(correct)/float(total)


    if val_acc > Best_value:
        print('Saving..')
        print("best_RAFWval_acc: %0.3f" % val_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
	        'best_RAFval_acc': val_acc,
    	    'best_RAFval_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,opt.savefile))
        Best_value = val_acc
        best_val_acc_epoch = epoch
    return val_loss

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    val_loss= val(epoch)

print("best_RAFval_acc: %0.3f" % Best_value)
print("best_RAFval_acc_epoch: %d" % best_val_acc_epoch)

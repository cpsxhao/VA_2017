#!/usr/bin/env python

# 采用20折交叉验证，将训练集1300划分为20个大小为65的子集，依然采用 Top5 准确率作为评估标准

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models
import train as tra
import evaluate as eva
from dataset import VideoFeatDataset as dset
from tools import utils

filelist = []
with open(eva.opt.video_flist, 'r') as rf:
    for line in rf.readlines():
        filepath = line.strip()
        filelist.append(filepath)
with open(tra.opt.flist, 'r') as rf:
    for line in rf.readlines():
        filepath = line.strip()
        filelist.append(filepath)

num = len(filelist)
testlist = []
trainlist = []

FoldNum = 20

for i in range(FoldNum):
    assert num%FoldNum == 0
    testlist.append(filelist[i*num/FoldNum:(i+1)*num/FoldNum])
    trainlist.append(filelist[0:i*num/FoldNum] + filelist[(i+1)*num/FoldNum:])

for iter in range(FoldNum):
    print('this is ' + str(iter) + 'th iter')
    
    # train
    # create model
    model = models.VGGLikeFC()

    if tra.opt.init_model != '':
        print('loading pretrained model from {0}'.format(tra.opt.init_model))
        model.load_state_dict(torch.load(tra.opt.init_model))

    # Contrastive Loss
    criterion = models.ContrastiveLoss()

    if tra.opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    train_dataset = dset(tra.opt.data_dir, flist=tra.opt.flist)
    train_dataset.pathlist = tra[iter]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tra.opt.batchSize,
                                              shuffle=True, num_workers=int(tra.opt.workers))

    print('number of train samples is: {0} % {1} times'.format(len(train_dataset),iter))
    print('finished loading train data % {0} times'.format(iter))
    
    # new optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay)

    for epoch in range(tra.opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        tra.train(train_loader, model, criterion, optimizer, epoch, tra.opt)
        optimizer.step()

        ##################################
        # save checkpoint every 10 epochs
        ##################################
        if ((epoch+1) % tra.opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_{3}times_state_epoch{2}.pth'.format(tra.opt.checkpoint_folder, tra.opt.prefix, epoch+1, iter+1)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

    # test
    test_video_dataset = dset(eva.opt.data_dir, eva.opt.video_flist, which_feat='vfeat')
    test_audio_dataset = dset(eva.opt.data_dir, eva.opt.audio_flist, which_feat='afeat')
    test_video_dataset.pathlist = testlist[iter];
    test_audio_dataset.pathlist = testlist[iter];
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset,
                                                    batch_size=eva.opt.batchSize,
                                                    shuffle=False, 
                                                    num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, 
                                                    batch_size=eva.opt.batchSize,
                                                    shuffle=False, 
                                                    num_workers=int(eva.opt.workers))

    print('number of test samples is: {0} % {1} times'.format(len(test_video_dataset),iter))
    print('finished loading test data % {0} times'.format(iter))

    if eva.opt.init_model != '':
        print('loading pretrained model from {0}'.format(eva.opt.init_model))
        model.load_state_dict(torch.load(eva.opt.init_model))

    if eva.opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()            
    
    evaluate.test(test_video_loader, test_audio_loader, model, opt)

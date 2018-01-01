#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models
import modelsLY
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils
from evaluate import testv2


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))

train_dataset = dset(opt.data_dir, flist=opt.flist)

print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    torch.manual_seed(opt.manualSeed)
else:
    if int(opt.ngpu) == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
            torch.cuda.manual_seed(opt.manualSeed)
            cudnn.benchmark = True
print('Random Seed: {0}'.format(opt.manualSeed))

# test options
parser_test = OptionParser()
parser_test.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts_test, args_test) = parser_test.parse_args()
assert isinstance(opts_test, object)
opt_test = Config(opts_test.config)

if opt_test.checkpoint_folder is None:
    opt_test.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(opt_test.data_dir, opt_test.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt_test.data_dir, opt_test.audio_flist, which_feat='afeat')

test_train_video_dataset = dset(opt_test.data_dir, opt_test.video_flist2, which_feat='vfeat')
test_train_audio_dataset = dset(opt_test.data_dir, opt_test.audio_flist2, which_feat='afeat')



# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('float32')
        target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2)

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target

        # transpose the feats
        #vfeat0 = vfeat0.transpose(2, 1)
        #afeat0 = afeat0.transpose(2, 1)

        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()

        # forward, backward optimize
        sim = model(vfeat_var, afeat_var)   # inference simialrity
        loss = criterion(sim, target_var)   # compute contrastive loss

        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat0.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)

def trainv2(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        '''
        for index in range(orders.shape[0] / 2 + 1 ):
            exchange_order = random.randint(0,orders.shape[0]);
            backup = orders[index];
            orders[index] = orders[exchange_order]
            orders[exchange_order] = backup;
        '''
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        wrong_pos = np.where(shuffle_orders == orders)
        if (wrong_pos[0].shape[0] != 0):
            pos_index = torch.from_numpy(wrong_pos[0])
            pos_index_new = torch.from_numpy(np.mod(wrong_pos[0]+1, bz))
            afeat2[pos_index, :, :] = afeat2[pos_index_new]

        # concat the vfeat and afeat respectively
        afeat0 = afeat
        vfeat0 = vfeat

        # generating the labels
        # 1. the labels for the shuffled feats
        #label1 = (orders == shuffle_orders + 0).astype('float32')
        #target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        #label2 = label1.copy()
        #label2[:] = 1
        #target2 = torch.from_numpy(label2)

        # concat the labels together
        #target = torch.cat((target2, target1), 0)
        #target = 1 - target

        # transpose the feats
        #vfeat0 = vfeat0.transpose(2, 1)
        #afeat0 = afeat0.transpose(2, 1)

        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        neg_afeatvar = Variable(afeat2)
        #target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            neg_afeatvar = neg_afeatvar.cuda()

        # forward, backward optimize
        # here model returns three vectors
        sim = model(vfeat_var, afeat_var,neg_afeatvar)   # inference simialrity
        #loss = criterion(sim, target_var)   # compute contrastive loss
        loss = criterion(sim[0],sim[1],sim[2])  # compute contrastive loss
        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat0.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)

def main():
    global opt
    global opt_test
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))

    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))

    test_train_video_loader = torch.utils.data.DataLoader(test_train_video_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))
    test_train_audio_loader = torch.utils.data.DataLoader(test_train_audio_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))

    # create model
    if opt.model == 'IBFC':
        model = models.ImageBasedFC()
    elif opt.model == 'HFFC':
        model = models.HighFramePoolFC()
    elif opt.model == 'LSTM':
        model = models.LSTMModel()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    #criterion = models.ContrastiveLoss()
    criterion = torch.nn.TripletMarginLoss(margin=opt.margin)
    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    #optimizer = optim.SGD(model.parameters(), opt.lr,
    #                            momentum=opt.momentum,
    #                            weight_decay=opt.weight_decay)

    # new optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr)

    best = 0


    # adjust learning rate every lr_decay_epoch
    #lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)   #poly policy
    #scheduler = LR_Policy(optimizer, lambda_lr)

    for epoch in range(opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        trainv2(train_loader, model, criterion, optimizer, epoch, opt)
        optimizer.step()

        ##################################
        # save checkpoint every 10 epochs
        ##################################
        if ((epoch+1) % opt.epoch_save) == 0:
            opt_test.topk = 5
            acc = testv2(test_video_loader, test_audio_loader, model, opt_test)
            if (acc >= best):
                path_checkpoint = '{0}/best_{1}_epoch{2}.pth'.format(opt.checkpoint_folder, acc, epoch + 1)
                utils.save_checkpoint(model.state_dict(), path_checkpoint)
                best = acc

            opt_test.topk = 1
            testv2(test_train_video_loader, test_train_audio_loader, model, opt_test)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

import models
import modelsLY
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils

from evaluate import testv2 as singleTest
from UnionEvaluate import testv2 as unionTest


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(opt.data_dir, opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
else:
    if int(opt.ngpu) == 1:
        print('so we use gpu 1 for testing')
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        cudnn.benchmark = True
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))


def main():
    global opt

    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))

    # create model
    model1 = models.ImageBasedFC()
    model2 = models.HighFramePoolFC()
    model3 = models.LSTMModel()

    if opt.init_model_IBFC != '':
        print('loading pretrained model from {0}'.format(opt.init_model_IBFC))
        model1.load_state_dict(torch.load(opt.init_model_IBFC))
        print('loading pretrained model from {0}'.format(opt.init_model_HFFC))
        model2.load_state_dict(torch.load(opt.init_model_HFFC))
        print('loading pretrained model from {0}'.format(opt.init_model_LSTM))
        model3.load_state_dict(torch.load(opt.init_model_LSTM))

    if opt.cuda:
        print('shift model to GPU .. ')
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()

    print('using LSTM Model to perfrom single test:')
    singleTest(test_video_loader, test_audio_loader, model3, opt)
    print('using 3 Models to perfrom union test:')
    unionTest(test_video_loader, test_audio_loader, model1, model2, model3, opt)


if __name__ == '__main__':
    main()
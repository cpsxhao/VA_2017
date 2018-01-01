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


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

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

# test function for metric learning
def test(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()

    sim_mat = []
    right = 0
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            # transpose feats
            #vfeat = vfeat.transpose(2,1)
            #afeat = afeat.transpose(2,1)

            # shuffling the index orders
            bz = vfeat.size()[0]
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                cur_sim = model.forward(vfeat_var, afeat_var)
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                if k in order:
                    right = right + 1
            print('The similarity matrix: \n {}'.format(simmat))
            print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/bz))
def testv2(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()

    sim_mat = []
    right = 0
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            # transpose feats
            #vfeat = vfeat.transpose(2,1)
            #afeat = afeat.transpose(2,1)

            # shuffling the index orders
            bz = vfeat.size()[0]
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()
                #print('dalong log : check input size ',vfeat_var.size(),afeat_var.size())
                cur_sim = model.forward(vfeat_var, afeat_var,afeat_var)
                #print('dalong log : check output size ', cur_sim[0].size())

                cur_sim = F.pairwise_distance(cur_sim[0],cur_sim[1])
                #print('dalong log : check sim size ', cur_sim.size());
                #exit();
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                if k in order:
                    right = right + 1
            print('The similarity matrix: \n {}'.format(simmat))
            print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/bz))
            return right/bz

def main():
    global opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))

    # create model
    model = models.ImageBasedFC()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()

    testv2(test_video_loader, test_audio_loader, model, opt)


if __name__ == '__main__':
    main()

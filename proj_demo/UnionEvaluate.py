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
def testv2(video_loader, audio_loader, model1, model2, model3, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model1.eval()
    model2.eval()
    model3.eval()

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

                cur_sim1 = model1.forward(vfeat_var, afeat_var,afeat_var)
                cur_sim1 = F.pairwise_distance(cur_sim1[0],cur_sim1[1])
                cur_sim1 = (cur_sim1 - torch.mean(cur_sim1))/torch.sqrt(torch.var(cur_sim1))

                cur_sim2 = model1.forward(vfeat_var, afeat_var, afeat_var)
                cur_sim2 = F.pairwise_distance(cur_sim2[0], cur_sim2[1])
                cur_sim2 = (cur_sim2 - torch.mean(cur_sim2)) / torch.sqrt(torch.var(cur_sim2))

                cur_sim3 = model1.forward(vfeat_var, afeat_var, afeat_var)
                cur_sim3 = F.pairwise_distance(cur_sim3[0], cur_sim3[1])
                cur_sim3 = (cur_sim3 - torch.mean(cur_sim3)) / torch.sqrt(torch.var(cur_sim3))

                cur_sim = 0.9*cur_sim1 + cur_sim2 + cur_sim3

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
    model1 = models.ImageBasedFC()
    model2 = models.HighFramePoolFC()
    model3 = models.LSTMModel()
    #model = modelsLY.VAMetric()
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

    testv2(test_video_loader, test_audio_loader, model1, model2, model3, opt)


if __name__ == '__main__':
    main()

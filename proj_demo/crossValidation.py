#!/usr/bin/env python

# 采用20折交叉验证，将训练集1300划分为20个大小为65的子集，依然采用 Top5 准确率作为评估标准

from train import train
from evaluate import test
from dataset import VideoFeatDataset as dset

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
    assert num % FoldNum == 0
    testlist.append(filelist[i * num / FoldNum:(i + 1) * num / FoldNum])
    trainlist.append(filelist[0:i * num / FoldNum] + filelist[(i + 1) * num / FoldNum:])

for iter in range(FoldNum):
    print('this is ' + str(iter) + 'th iter')

    train_dataset = dset(tra.opt.data_dir, flist=tra.opt.flist)
    train_dataset.pathlist = trainlist[iter]

    print('number of train samples is: {0} % {1} times'.format(len(train_dataset), iter))
    print('finished loading train data % {0} times'.format(iter))

    test_video_dataset = dset(eva.opt.data_dir, eva.opt.video_flist, which_feat='vfeat')
    test_audio_dataset = dset(eva.opt.data_dir, eva.opt.audio_flist, which_feat='afeat')
    test_video_dataset.pathlist = testlist[iter];
    test_audio_dataset.pathlist = testlist[iter];

    print('number of test samples is: {0} % {1} times'.format(len(test_video_dataset), iter))
    print('finished loading test data % {0} times'.format(iter))

    global opt
    # create model
    model = models.VGGLikeFC()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    criterion = models.ContrastiveLoss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # new optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay)

    for epoch in range(opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt)
        optimizer.step()

        ##################################
        # save checkpoint every 10 epochs
        ##################################
        if ((epoch + 1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_{3}times_state_epoch{2}.pth'.format(opt.checkpoint_folder, opt.prefix, epoch + 1,
                                                                           i + 1)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

    test(test_video_loader, test_audio_loader, model, opt)

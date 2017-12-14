#!/usr/bin/env python

# 采用20折交叉验证，将训练集1300划分为20个大小为65的子集，依然采用 Top5 准确率作为评估标准

import train as tra
import evaluate as eva

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
for i in range(20):
    testlist.append(filelist[i*num/20:(i+1)*num/20])
    trainlist.append(filelist[0:i*num/20] + filelist[(i+1)*num/20:])

for iter in range(20):
    print('this is ' + str(iter) + 'th iter')

    train_dataset = dset(tra.opt.data_dir, flist=tra.opt.flist)
    train_dataset.pathlist = trainlist[iter]

    print('number of train samples is: {0}'.format(len(train_dataset)))
    print('finished loading data')

    test_video_dataset = dset(eva.opt.data_dir, eva.opt.video_flist, which_feat='vfeat')
    test_audio_dataset = dset(eva.opt.data_dir, eva.opt.audio_flist, which_feat='afeat')
    test_video_dataset.pathlist = testlist[iter];
    test_audio_dataset.pathlist = testlist[iter];

    print('number of test samples is: {0}'.format(len(test_video_dataset)))
    print('finished loading data')

    tra.main()
    eva.main()
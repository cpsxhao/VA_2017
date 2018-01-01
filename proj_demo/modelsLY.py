import torch
import torch.nn as nn
from   torch.autograd import Variable
import torch.nn.functional as F
import pdb

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.GRU(input_size, hidden_size, cell_num, batch_first=True)

    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        #c0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()
            #c0 = c0.cuda()

        # aggregated feature
        # feat, _ = self.rnn(feats, (h0, c0))
        feat, _ = self.rnn(feats, h0)
        return feat


# Visual-audio multimodal metric learning: 
# MaxPool + FC ---> Conv1d + AvgPool + FC
class LSTMModel(nn.Module):
    def __init__(self, framenum=120):
        super(LSTMModel, self).__init__()
        self.Vlstm = FeatAggregate(input_size=1024, hidden_size=256)
        self.Alstm = FeatAggregate(input_size=128, hidden_size=128)
        self.Vcnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(120, 256), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.Acnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(120, 128), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
                
    def forward(self, vfeat, afeat, neg_afeat):
        #vfeat 64*120*1024
        #afeat 64*120*128
        vfeat = self.Vlstm(vfeat)        
        vfeat = self.Vcnn1(vfeat.unsqueeze(1))
        vfeat = vfeat.squeeze()
        vfeat = self.fc(vfeat)

        afeat = self.Alstm(afeat)
        afeat = self.Acnn1(afeat.unsqueeze(1))
        afeat = afeat.squeeze()
        afeat = self.fc(afeat)

        neg_afeat = self.Alstm(neg_afeat)
        neg_afeat = self.Acnn1(neg_afeat.unsqueeze(1))
        neg_afeat = neg_afeat.squeeze()
        neg_afeat = self.fc(neg_afeat)


        return vfeat, afeat, neg_afeat

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=0.9):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


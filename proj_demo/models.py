import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, out_size)

    def forward(self, feats):
        h_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        c_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        h_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)
        c_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)

        if feats.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
            h_t2 = h_t2.cuda()
            c_t2 = c_t2.cuda()

        for _, feat_t in enumerate(feats.chunk(feats.size(1), dim=1)):
            h_t, c_t = self.lstm1(feat_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

        # aggregated feature
        feat = h_t2
        return feat

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 512, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)


    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = self.vfc(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


class VGGLikeFC(nn.Module):
    def __init__(self, framenum = 120):
        super(VGGLikeFC, self).__init__()

        self.poolf1 = nn.AdaptiveAvgPool1d(60)
        #self.poolf2 = nn.AdaptiveMaxPool1d(30)

        self.vModules = nn.Sequential(
            nn.AdaptiveAvgPool1d(512),
            nn.BatchNorm1d(60),
            nn.Linear(512, 96),
            nn.BatchNorm1d(60),
            nn.Dropout(0.6),
            nn.ReLU(),

            nn.Linear(96, 32),
            nn.BatchNorm1d(60),
            nn.Dropout(0.6),
            nn.ReLU()
        )


        self.aModules = nn.Sequential(
            nn.BatchNorm1d(60),
            nn.Linear(128, 96),
            nn.BatchNorm1d(60),
            nn.Dropout(0.6),
            nn.ReLU(),

            nn.Linear(96,32),
            nn.BatchNorm1d(60),
            nn.Dropout(0.6),
            nn.ReLU()
        )

        self.deciModule = nn.Sequential(
            nn.Linear(32,1)
        )

        self.pools1 = nn.AvgPool1d(32)
        self.pools2 = nn.MaxPool1d(32)


        self.dist = nn.PairwiseDistance(p=2)

        def init_params(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform(m.weight)
                    nn.init.constant(m.bias, 0)
                elif isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform(m.weight)
                    nn.init.constant(m.bias, 0)
    def forward(self, vfeat, afeat):
        vfeat = vfeat.transpose(1,2)
        afeat = afeat.transpose(1,2)

        vfeat = self.poolf1(vfeat)
        afeat = self.poolf1(afeat)

        vfeat = vfeat.transpose(1,2)
        afeat = afeat.transpose(1,2)
        vfeat = vfeat.contiguous()
        afeat = afeat.contiguous()

        vfeat = self.vModules(vfeat)
        afeat = self.aModules(afeat)

        #vfeat = self.pools1(vfeat)
        #afeat = self.pools1(afeat)
        vfeat = self.deciModule(vfeat)
        afeat = self.deciModule(afeat)

        vfeat = vfeat.contiguous()
        afeat = afeat.contiguous()
        vfeat = vfeat.view(-1, 60)
        afeat = afeat.view(-1, 60)

        dist = self.dist(vfeat, afeat)

        return dist



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        dist = dist.view(-1)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        #loss = torch.mean((1 - label) * dist +
        #        (label) * torch.clamp(self.margin - dist, min=0.0))
        return loss


class L2Normalize(torch.nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, feats):
        for featNum in range(0, feats.size(0)):
            for frame in range(0, feats.size(1)):
                feats[featNum, frame, :] = feats[featNum, frame, :].clone() / torch.norm(feats[featNum, frame, :].clone(), 2)
        return feats

class frameMeanDistance(torch.nn.Module):
    def __init__(self):
        super(frameMeanDistance, self).__init__()

    def forward(self, vfeats, afeats):
        dist = F.pairwise_distance(vfeats[:,0,:], afeats[:,0,:], p=2)
        for frame in range(1, vfeats.size(1)):
            dist = dist + F.pairwise_distance(vfeats[:, frame, :], afeats[:, frame, :], p=2)
        dist = dist / vfeats.size(1)
        return dist
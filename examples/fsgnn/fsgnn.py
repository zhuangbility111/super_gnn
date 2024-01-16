import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class FSGNN(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout,aggr_type='cat'):
        super(FSGNN,self).__init__()
        if aggr_type == 'cat':
            self.fc2 = nn.Linear(nhidden*nlayers,nclass)
        elif aggr_type == 'sum':
            self.fc2 = nn.Linear(nhidden,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        # self.att = nn.Parameter(torch.ones(nnodes, nlayers))
        # self.sm = nn.Softmax(dim=1)
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.n_hidden = nhidden
        self.aggr_type = aggr_type


    def forward(self,list_mat,layer_norm):
        mask = self.sm(self.att)
        device = list_mat[0].get_device()
        if self.aggr_type == 'cat':
            final_mat = torch.tensor([])
        elif self.aggr_type == 'sum':
            final_mat = torch.zeros(list_mat[0].shape[0], self.n_hidden)
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)
            # tmp_out = torch.mul(mask[:, ind].view(-1, 1),tmp_out)

            if self.aggr_type == 'cat':
                final_mat = torch.cat((final_mat,tmp_out),dim=1)
            elif self.aggr_type == 'sum':
                final_mat += tmp_out
            # list_out.append(tmp_out)

        # final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)


        return F.log_softmax(out, dim=1)

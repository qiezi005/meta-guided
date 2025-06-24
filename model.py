"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
import numpy as np
import os
import math
from torch.nn import Linear, BatchNorm1d, Conv2d, Sequential, ReLU, Softmax, MultiheadAttention
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GPSConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import remove_self_loops


class MT_GAT_topk_shareEn_multiple(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, hc5, hc6, ratio):
        super(MT_GAT_topk_shareEn_multiple, self).__init__()
        torch.manual_seed(12345)
        self.conv0 = GATConv(263, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels, hc2)
        self.lin13 = Linear(math.ceil(ratio*264)*hidden_channels, hc3)
        self.lin14 = Linear(math.ceil(ratio*264)*hidden_channels, hc4)
        self.lin15 = Linear(math.ceil(ratio*264)*hidden_channels, hc5)
        self.lin16 = Linear(math.ceil(ratio*264)*hidden_channels, hc6)
        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 1)
        self.lin33 = Linear(hc3, 1)
        self.lin34 = Linear(hc4, 1)
        self.lin35 = Linear(hc5, 1)
        self.lin36 = Linear(hc6, 1)
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool3 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool4 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool5 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool6 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        x = self.conv0(x, edge_index, edge_weight)
        x = F.relu(x)

        x11 = x
        edge_index11 = edge_index
        edge_weight11 = edge_weight
        batch11 = batch

        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        edge_weight22 = edge_weight
        batch22 = batch
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)

        #MLP for joint information


        

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x11
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch

        x3 = x11
        edge_index3 = edge_index
        edge_weight3 = edge_weight
        batch3 = batch

        x4 = x22
        edge_index4 = edge_index
        edge_weight4 = edge_weight
        batch4 = batch

        x5 = x22
        edge_index5 = edge_index
        edge_weight5 = edge_weight
        batch5 = batch

        x6 = x22
        edge_index6 = edge_index
        edge_weight6 = edge_weight
        batch6 = batch

        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device) 
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

       
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)   
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))
        #print(x2)

        x3, edge_index3, edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index3,edge_weight3, batch3)
        x3 = self.flatten(x3, batch3)
        x3 = x3.to(device)   
        x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.lin13(F.relu(x3))
        x3 = self.lin33(F.relu(x3))
        #print(x2)

        x4, edge_index4, edge_attr4, batch4, perm4, score4 = self.pool4(x4, edge_index4, edge_weight4, batch4)
        x4 = self.flatten(x4, batch4)
        x4 = x4.to(device)   
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x4 = self.lin14(F.relu(x4))
        x4 = self.lin34(F.relu(x4))

        x5, edge_index5, edge_attr5, batch5, perm5, score5 = self.pool5(x5, edge_index5, edge_weight5, batch5)
        x5 = self.flatten(x5, batch5)
        x5 = x5.to(device)   
        x5 = F.dropout(x5, p=0.5, training=self.training)
        x5 = self.lin15(F.relu(x5))
        x5 = self.lin35(F.relu(x5))

        x6, edge_index6, edge_attr6, batch6, perm6, score6 = self.pool6(x6, edge_index6, edge_weight6, batch6)
        x6 = self.flatten(x6, batch6)
        x6 = x6.to(device)   
        x6 = F.dropout(x6, p=0.5, training=self.training)
        x6 = self.lin16(F.relu(x6))
        x6 = self.lin36(F.relu(x6))
       
        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), x3, perm3, torch.sigmoid(score3), \
        x4, perm4, torch.sigmoid(score4), x5, perm5, torch.sigmoid(score5), x6, perm6, torch.sigmoid(score6)

class MT_GAT_topk_shareEn_multiple8(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, hc5, hc6, hc7, hc8, ratio):
        super(MT_GAT_topk_shareEn_multiple8, self).__init__()
        torch.manual_seed(12345)
        self.conv0 = GATConv(263, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        #hidden_channels = hidden_channels + 1
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool3 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool4 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool5 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool6 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool7 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool8 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels, hc2)
        self.lin13 = Linear(math.ceil(ratio*264)*hidden_channels, hc3)
        self.lin14 = Linear(math.ceil(ratio*264)*hidden_channels, hc4)
        self.lin15 = Linear(math.ceil(ratio*264)*hidden_channels, hc5)
        self.lin16 = Linear(math.ceil(ratio*264)*hidden_channels, hc6)
        self.lin17 = Linear(math.ceil(ratio*264)*hidden_channels, hc7)
        self.lin18 = Linear(math.ceil(ratio*264)*hidden_channels, hc8)
        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 1)
        self.lin33 = Linear(hc3, 1)
        self.lin34 = Linear(hc4, 1)
        self.lin35 = Linear(hc5, 1)
        self.lin36 = Linear(hc6, 1)
        self.lin37 = Linear(hc7, 1)
        self.lin38 = Linear(hc8, 1)
        
        

        self.softmax_func=Softmax(dim=1)
        self.lin4 = Linear(264, 66)      
        self.lin5 = Linear(66, 264)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        x = self.conv0(x, edge_index, edge_weight)
        x = F.relu(x)

        x11 = x
        edge_index11 = edge_index
        edge_weight11 = edge_weight
        batch11 = batch

        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        edge_weight22 = edge_weight
        batch22 = batch
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
     
        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x11
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch

        x3 = x11
        edge_index3 = edge_index
        edge_weight3 = edge_weight
        batch3 = batch

        x4 = x11
        edge_index4 = edge_index
        edge_weight4 = edge_weight
        batch4 = batch

        x5 = x22
        edge_index5 = edge_index
        edge_weight5 = edge_weight
        batch5 = batch

        x6 = x22
        edge_index6 = edge_index
        edge_weight6 = edge_weight
        batch6 = batch

        x7 = x22
        edge_index7 = edge_index
        edge_weight7 = edge_weight
        batch7 = batch

        x8 = x22
        edge_index8 = edge_index
        edge_weight8 = edge_weight
        batch8 = batch

        #x1 = torch.cat((x1, all), 1)
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        all1 = self.flatten(x1, batch)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

        #x2 = torch.cat((x2,all),1)
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        all1 = self.flatten(x2, batch2)
        x2 = x2.to(device)   
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))
        #print(x2)
        x3, edge_index3, edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index3,edge_weight3, batch3)
        x3 = self.flatten(x3, batch3)
        x3 = x3.to(device) 
        x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.lin13(F.relu(x3))
        x3 = self.lin33(F.relu(x3))
        #print(x2)
        x4, edge_index4, edge_attr4, batch4, perm4, score4 = self.pool4(x4, edge_index4, edge_weight4, batch4)
        x4 = self.flatten(x4, batch4)
        x4 = x4.to(device)   
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x4 = self.lin14(F.relu(x4))
        x4 = self.lin34(F.relu(x4))

        x5, edge_index5, edge_attr5, batch5, perm5, score5 = self.pool5(x5, edge_index5, edge_weight5, batch5)
        x5 = self.flatten(x5, batch5)
        x5 = x5.to(device) 
        x5 = F.dropout(x5, p=0.5, training=self.training)
        x5 = self.lin15(F.relu(x5))
        x5 = self.lin35(F.relu(x5))


        x6, edge_index6, edge_attr6, batch6, perm6, score6 = self.pool6(x6, edge_index6, edge_weight6, batch6)
        x6 = self.flatten(x6, batch6)
        x6 = x6.to(device)  
       # x6 = torch.cat((x6,all),1) 
        x6 = F.dropout(x6, p=0.5, training=self.training)
        x6 = self.lin16(F.relu(x6))
        x6 = self.lin36(F.relu(x6))


        x7, edge_index7, edge_attr7, batch7, perm7, score7 = self.pool7(x7, edge_index7, edge_weight7, batch7)
        x7 = self.flatten(x7, batch7)
        x7 = x7.to(device)  
        x7 = F.dropout(x7, p=0.5, training=self.training)
        x7 = self.lin17(F.relu(x7))
        x7 = self.lin37(F.relu(x7))

        x8, edge_index8, edge_attr8, batch8, perm8, score8 = self.pool8(x8, edge_index8, edge_weight8, batch8)
        x8 = self.flatten(x8, batch8)
        x8 = x8.to(device)    
        x8 = F.dropout(x8, p=0.5, training=self.training)
        x8 = self.lin18(F.relu(x8))
        x8 = self.lin38(F.relu(x8))
       
        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), x3, perm3, torch.sigmoid(score3), \
        x4, perm4, torch.sigmoid(score4), x5, perm5, torch.sigmoid(score5), x6, perm6, torch.sigmoid(score6), x7, perm7, torch.sigmoid(score7), x8, perm8, torch.sigmoid(score8)

class MT_GAT_topk_shareEn_multiple8_joint(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, hc5, hc6, hc7, hc8, ratio):
        super(MT_GAT_topk_shareEn_multiple8_joint, self).__init__()
        torch.manual_seed(12345)
        #self.conv0 = GATConv(263, hidden_channels)
        hidden_channels3 = 264
        nn = Sequential(
                Linear(hidden_channels3, hidden_channels3),
                ReLU(),
                Linear(hidden_channels3, hidden_channels3),
            )
        self.conv0 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)

        #nn = Sequential(
        #        Linear(hidden_channels, hidden_channels),
        #        ReLU(),
        #        Linear(hidden_channels, hidden_channels),
        #    )
        #self.conv2 = GPSConv(hidden_channels, GINEConv(nn, edge_dim = 1), heads=2)

        self.conv1 = GATConv(hidden_channels3, hidden_channels)
        self.conv2 = GATConv(hidden_channels3, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool3 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool4 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool5 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool6 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool7 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool8 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc2)
        self.lin13 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc3)
        self.lin14 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc4)
        self.lin15 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc5)
        self.lin16 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc6)
        self.lin17 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc7)
        self.lin18 = Linear(math.ceil(ratio*264)*hidden_channels+264*hidden_channels, hc8)
        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 1)
        self.lin33 = Linear(hc3, 1)
        self.lin34 = Linear(hc4, 1)
        self.lin35 = Linear(hc5, 1)
        self.lin36 = Linear(hc6, 1)
        self.lin37 = Linear(hc7, 1)
        self.lin38 = Linear(hc8, 1)
        
        

        self.softmax_func=Softmax(dim=1)
        self.lin4 = Linear(264, 66)      
        self.lin5 = Linear(66, 264)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        edge_weight00 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight00)
        x = F.relu(x)
        #print(x.shape)
        x11 = x
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        batch11 = batch

        #x11 = self.conv1(x11, edge_index11, batch11, edge_attr = edge_weight11)
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        batch22 = batch
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight22 = edge_weight
        #x22 = self.conv2(x22, edge_index22, batch22, edge_attr = edge_weight22)
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
        
        
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all = torch.cat((x11, x22),1)
        all = torch.reshape(all, ((len(torch.unique(batch))), 1, seg, all.shape[1])) ## can be really reshaped?
        all = all.to(device)
        all = self.conv4(all)
        all = torch.reshape(all, ((len(torch.unique(batch))), seg))
        att = self.lin4(all)
        att = F.relu(att)
        att = self.lin5(att)
        #att = F.relu(att)
        att = self.softmax_func(att)
        att = torch.reshape(att, ((len(torch.unique(batch)))*seg, 1))
        hidden_channels = x11.shape[1]
        att = torch.tile(att, (1, hidden_channels))
        

        all1 = torch.mul(att, x11)
        all2 = torch.mul(att, x22)
        all1 = self.flatten(all1, batch)
        all2 = self.flatten(all2, batch)
        all1 = all1.to(device)
        all2 = all2.to(device)

        #print(all1.shape)
        
        #all = torch.reshape(all,(all.shape[0]*all.shape[1], 1))
        

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x11
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch

        x3 = x11
        edge_index3 = edge_index
        edge_weight3 = edge_weight
        batch3 = batch

        x4 = x11
        edge_index4 = edge_index
        edge_weight4 = edge_weight
        batch4 = batch

        x5 = x22
        edge_index5 = edge_index
        edge_weight5 = edge_weight
        batch5 = batch

        x6 = x22
        edge_index6 = edge_index
        edge_weight6 = edge_weight
        batch6 = batch

        x7 = x22
        edge_index7 = edge_index
        edge_weight7 = edge_weight
        batch7 = batch

        x8 = x22
        edge_index8 = edge_index
        edge_weight8 = edge_weight
        batch8 = batch

        #x1 = torch.cat((x1, all), 1)
        print(x1.shape)
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        x1 = torch.cat((x1, all2),1)
        print(x1.shape)
        print(all2.shape)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

        #x2 = torch.cat((x2,all),1)
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)  
        x2 = torch.cat((x2, all2),1) 
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))

        x3, edge_index3, edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index3,edge_weight3, batch3)
        x3 = self.flatten(x3, batch3)
        x3 = x3.to(device) 
        x3 = torch.cat((x3, all2),1)
        x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.lin13(F.relu(x3))
        x3 = self.lin33(F.relu(x3))


        x4, edge_index4, edge_attr4, batch4, perm4, score4 = self.pool4(x4, edge_index4, edge_weight4, batch4)
        x4 = self.flatten(x4, batch4)
        x4 = x4.to(device)  
        x4 = torch.cat((x4, all2),1)
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x4 = self.lin14(F.relu(x4))
        x4 = self.lin34(F.relu(x4))


        x5, edge_index5, edge_attr5, batch5, perm5, score5 = self.pool5(x5, edge_index5, edge_weight5, batch5)
        x5 = self.flatten(x5, batch5)
        x5 = x5.to(device) 
        x5 = torch.cat((x5, all1),1) 
        x5 = F.dropout(x5, p=0.5, training=self.training)
        x5 = self.lin15(F.relu(x5))
        x5 = self.lin35(F.relu(x5))

        x6, edge_index6, edge_attr6, batch6, perm6, score6 = self.pool6(x6, edge_index6, edge_weight6, batch6)
        x6 = self.flatten(x6, batch6)
        x6 = x6.to(device)  
        x6 = torch.cat((x6,all1),1) 
        x6 = F.dropout(x6, p=0.5, training=self.training)
        x6 = self.lin16(F.relu(x6))
        x6 = self.lin36(F.relu(x6))

        x7, edge_index7, edge_attr7, batch7, perm7, score7 = self.pool7(x7, edge_index7, edge_weight7, batch7)
        x7 = self.flatten(x7, batch7)
        x7 = x7.to(device)  
        x7 = torch.cat((x7, all1),1)
        x7 = F.dropout(x7, p=0.5, training=self.training)
        x7 = self.lin17(F.relu(x7))
        x7 = self.lin37(F.relu(x7))

        x8, edge_index8, edge_attr8, batch8, perm8, score8 = self.pool8(x8, edge_index8, edge_weight8, batch8)
        x8 = self.flatten(x8, batch8)
        x8 = x8.to(device)   
        x8 = torch.cat((x8, all1),1)
        x8 = F.dropout(x8, p=0.5, training=self.training)
        x8 = self.lin18(F.relu(x8))
        x8 = self.lin38(F.relu(x8))
       
        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), x3, perm3, torch.sigmoid(score3), \
        x4, perm4, torch.sigmoid(score4), x5, perm5, torch.sigmoid(score5), x6, perm6, torch.sigmoid(score6), x7, perm7, torch.sigmoid(score7), x8, perm8, torch.sigmoid(score8)

class MT_GAT_topk_shareEn_multiple8_joint_last(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, ratio):
        super(MT_GAT_topk_shareEn_multiple8_joint_last, self).__init__()
        torch.manual_seed(12345)
        #self.conv0 = GATConv(263, hidden_channels)
        hidden_channels3 = 264
        nn = Sequential(
               Linear(hidden_channels3, hidden_channels3),
               ReLU(),
               Linear(hidden_channels3, hidden_channels3),
           )
        self.conv0 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)
       # self.conv1 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)

        #nn2 = Sequential(
        #        Linear(hidden_channels3, hidden_channels3),
        #        ReLU(),
        #        Linear(hidden_channels3, hidden_channels3),
        #    )
        #self.conv2 = GPSConv(hidden_channels3, GINEConv(nn2, edge_dim = 1), heads=2)
        
        
        #nn = Sequential(
        #        Linear(hidden_channels, hidden_channels),
        #        ReLU(),
        #        Linear(hidden_channels, hidden_channels),
        #    )
        #self.conv2 = GPSConv(hidden_channels, GINEConv(nn, edge_dim = 1), heads=2)
       # hidden_channels = 264
        self.conv1 = GATConv(hidden_channels3, hidden_channels)
        self.conv2 = GATConv(hidden_channels3, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels, hc2)
       
        #self.lin1 = Linear(hidden_channels*hidden_channels, hc)
        #self.lin12 = Linear(hidden_channels*hidden_channels, hc2)

        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 2)
        
        
        # self.lin111 = Linear(264*hidden_channels, hc)
        # self.lin1111 = Linear(hc, 1)

        # self.lin222 = Linear(264*hidden_channels, hc)
        # self.lin2222 = Linear(hc, 1)
    
        self.softmax_func=Softmax(dim=1)
        self.lin4 = Linear(264, 66)      
        self.lin5 = Linear(66, 264)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        edge_weight00 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight00)
        x = F.relu(x)

        x11 = x
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        #x11 = self.conv1(x11, edge_index11, batch, edge_attr = edge_weight11)
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1)) #edge_index
        edge_weight22 = edge_weight
        #x22 = self.conv2(x22, edge_index22, batch, edge_attr = edge_weight22)
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all = torch.cat((x11, x22),1) #2640*128
        all = torch.reshape(all, ((len(torch.unique(batch))), 1, seg, all.shape[1])) ## can be really reshaped?
        all = all.to(device) #10*1*264*128
        all = self.conv4(all) #10*1*264*1
        all = torch.reshape(all, ((len(torch.unique(batch))), seg)) #10*264
        att = self.lin4(all)
        att = F.relu(att)
        att = self.lin5(att)  #10*264
        att = torch.sigmoid(att)
        att = torch.reshape(att, ((len(torch.unique(batch)))*seg, 1)) #(10*264)*1
        hidden_channels = x11.shape[1]
        attt = torch.tile(att, (1, hidden_channels))

        all1 = torch.mul(attt, x11)
        all2 = torch.mul(attt, x22)
        all1 = all1.to(device)
        all2 = all2.to(device)

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x22
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch


        #x1 = torch.cat((x1, all), 1)
        
        x1 = x1 + all2
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

        x2 = x2 + all1
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)  
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))
        x2 = self.softmax_func(x2)

        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), att#, x111, x222, edge_sz, weight_sz, edge_cog, weight_cog

class MT_GAT_topk_shareEn_multiple8_joint_cross(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, ratio1, ratio2):
        super(MT_GAT_topk_shareEn_multiple8_joint_cross, self).__init__()
        torch.manual_seed(12345)
        #self.conv0 = GATConv(263, hidden_channels)
        hidden_channels3 = 264
        nn = Sequential(
               Linear(hidden_channels3, hidden_channels3),
               ReLU(),
               Linear(hidden_channels3, hidden_channels3),
           )
        self.conv0 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)
      
        #hidden_channels = 264
        self.conv1 = GATConv(hidden_channels3, hidden_channels)
        self.conv2 = GATConv(hidden_channels3, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        self.multihead_attn = MultiheadAttention(hidden_channels, 8,  batch_first='True')
        self.multihead_attn2 = MultiheadAttention(hidden_channels, 8,  batch_first='True')

        
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio1, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio2, multiplier=1, nonlinearity=torch.sigmoid)
        
        self.lin1 = Linear(math.ceil(ratio1*264)*hidden_channels, hc) #+264*hidden_channels
        self.lin12 = Linear(math.ceil(ratio2*264)*hidden_channels, hc2) #+264*hidden_channels
       
        #self.lin1 = Linear(hidden_channels*hidden_channels, hc)
        #self.lin12 = Linear(hidden_channels*hidden_channels, hc2)

        self.lin3 = Linear(hc, hc)
        self.lin32 = Linear(hc2, hc2)
        
        
        # self.lin111 = Linear(264*hidden_channels, hc)
        # self.lin1111 = Linear(hc, 1)

        # self.lin222 = Linear(264*hidden_channels, hc)
        # self.lin2222 = Linear(hc, 1)
    
        self.softmax_func=Softmax(dim=1)
        self.lin4 = Linear(hc, 1)
        self.lin42 = Linear(hc2, 2)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        edge_weight00 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight00)
        x = F.relu(x)

        x11 = x
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        #x11 = self.conv1(x11, edge_index11, batch, edge_attr = edge_weight11)
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1)) #edge_index
        edge_weight22 = edge_weight
        #x22 = self.conv2(x22, edge_index22, batch, edge_attr = edge_weight22)
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all11 = torch.reshape(x11, ((len(torch.unique(batch))), seg, x11.shape[1])) ## can be really reshaped?
        #print(all11.shape)
        all22 = torch.reshape(x22, ((len(torch.unique(batch))), seg, x22.shape[1]))

        all1, att = self.multihead_attn(all11, all22, all22)
        all2, att2 = self.multihead_attn2(all22, all11, all11)
        all1 = torch.reshape(all1, ((len(torch.unique(batch)))*seg, all1.shape[2]))
        all2 = torch.reshape(all2, ((len(torch.unique(batch)))*seg, all2.shape[2]))
        #print(att.shape)
        #print(x11.shape)
        # all = all.to(device) #10*1*264*128
        # all = self.conv4(all) #10*1*264*1
        # all = torch.reshape(all, ((len(torch.unique(batch))), seg)) #10*264
        # att = self.lin4(all)
        # att = F.relu(att)
        # att = self.lin5(att)  #10*264
        # att = torch.sigmoid(att)
        # att = torch.reshape(att, ((len(torch.unique(batch)))*seg, 1)) #(10*264)*1
        # hidden_channels = x11.shape[1]
        # attt = torch.tile(att, (1, hidden_channels))

        # all1 = torch.mul(attt, x11)
        # all2 = torch.mul(attt, x22)
        # all1 = all1.to(device)
        # all2 = all2.to(device)

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x22
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch


        #x1 = torch.cat((x1, all), 1)
        
        x1 = all1 + x1
        x1 = x1
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        #x1 = torch.cat((x1, all2), dim=0)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        #x1 = self.lin3(F.relu(x1))
        x1 = self.lin4(F.relu(x1))
        #print(x1)

        x2 = x2
        x2 = all1 + x2
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        #x2 = torch.cat((x2, all1), dim=0)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)  
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        #x2 = self.lin32(F.relu(x2))
        x2 = self.lin42(F.relu(x2))
        x2 = self.softmax_func(x2)

        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), att

class MT_GAT_topk_shareEn_multiple8_joint_sin(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, ratio):
        super(MT_GAT_topk_shareEn_multiple8_joint_sin, self).__init__()
        torch.manual_seed(12345)
        #self.conv0 = GATConv(263, hidden_channels)
        hidden_channels3 = 264
        nn = Sequential(
               Linear(hidden_channels3, hidden_channels3),
               ReLU(),
               Linear(hidden_channels3, hidden_channels3),
           )
        self.conv0 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)
      
        #hidden_channels = 264
        self.conv1 = GATConv(hidden_channels3, hidden_channels)
        self.conv2 = GATConv(hidden_channels3, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        self.multihead_attn = MultiheadAttention(hidden_channels, 4,  batch_first='True')

        
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels, hc2)
       
        #self.lin1 = Linear(hidden_channels*hidden_channels, hc)
        #self.lin12 = Linear(hidden_channels*hidden_channels, hc2)

        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 2)
        
        
        # self.lin111 = Linear(264*hidden_channels, hc)
        # self.lin1111 = Linear(hc, 1)

        # self.lin222 = Linear(264*hidden_channels, hc)
        # self.lin2222 = Linear(hc, 1)
    
        self.lin4 = Linear(264, 66)      
        self.lin5 = Linear(66, 264)
    
        self.softmax_func=Softmax(dim=1)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):
        edge_weight00 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight00)
        x = F.relu(x)

        x11 = x
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        x11 = F.relu(x11)

        x22 = x
        edge_index22 = edge_index
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1)) #edge_index
        edge_weight22 = edge_weight
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all = torch.cat((x11, x22),1) #2640*128
        all = torch.reshape(all, ((len(torch.unique(batch))), 1, seg, all.shape[1])) ## can be really reshaped?
        all = all.to(device) #10*1*264*128
        all = self.conv4(all) #10*1*264*1
        all = torch.reshape(all, ((len(torch.unique(batch))), seg)) #10*264
        att = self.lin4(all)
        att = F.relu(att)
        att = self.lin5(att)  #10*264
        att = torch.sigmoid(att)
        att = torch.reshape(att, ((len(torch.unique(batch)))*seg, 1)) #(10*264)*1
        hidden_channels = x11.shape[1]
        attt = torch.tile(att, (1, hidden_channels))

        all1 = torch.mul(attt, x11)
        all2 = torch.mul(attt, x22)
        all1 = all1.to(device)
        all2 = all2.to(device)

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x22
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch


        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)  
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))
        x2 = self.softmax_func(x2)

        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), att

class MT_GAT_topk_shareEn_multiple8_joint_last_GCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, hc5, hc6, hc7, hc8, ratio):
        super(MT_GAT_topk_shareEn_multiple8_joint_last_GCN, self).__init__()
        torch.manual_seed(12345)
        #self.conv0 = GATConv(263, hidden_channels)
        hidden_channels3 = 264
        nn = Sequential(
                Linear(hidden_channels3, hidden_channels3),
                ReLU(),
                Linear(hidden_channels3, hidden_channels3),
            )
        self.conv0 = GPSConv(hidden_channels3, GINEConv(nn, edge_dim = 1), heads=2)

        #nn = Sequential(
        #        Linear(hidden_channels, hidden_channels),
        #        ReLU(),
        #        Linear(hidden_channels, hidden_channels),
        #    )
        #self.conv2 = GPSConv(hidden_channels, GINEConv(nn, edge_dim = 1), heads=2)

        self.conv1 = GCNConv(hidden_channels3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels3, hidden_channels)
        self.conv4 = Conv2d(1,1, (1,2*hidden_channels))

        
        self.pool1 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool3 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool4 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool5 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool6 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool7 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool8 = TopKPooling(hidden_channels, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.lin1 = Linear(math.ceil(ratio*264)*hidden_channels, hc)
        self.lin12 = Linear(math.ceil(ratio*264)*hidden_channels, hc2)
        self.lin13 = Linear(math.ceil(ratio*264)*hidden_channels, hc3)
        self.lin14 = Linear(math.ceil(ratio*264)*hidden_channels, hc4)
        self.lin15 = Linear(math.ceil(ratio*264)*hidden_channels, hc5)
        self.lin16 = Linear(math.ceil(ratio*264)*hidden_channels, hc6)
        self.lin17 = Linear(math.ceil(ratio*264)*hidden_channels, hc7)
        self.lin18 = Linear(math.ceil(ratio*264)*hidden_channels, hc8)
        self.lin3 = Linear(hc, 1)
        self.lin32 = Linear(hc2, 1)
        self.lin33 = Linear(hc3, 1)
        self.lin34 = Linear(hc4, 1)
        self.lin35 = Linear(hc5, 1)
        self.lin36 = Linear(hc6, 1)
        self.lin37 = Linear(hc7, 1)
        self.lin38 = Linear(hc8, 1)
        
        self.lin111 = Linear(264*hidden_channels, hc)
        self.lin1111 = Linear(hc, 1)

        self.lin222 = Linear(264*hidden_channels, hc)
        self.lin2222 = Linear(hc, 1)
    
        self.softmax_func=Softmax(dim=1)
        self.lin4 = Linear(264, 66)      
        self.lin5 = Linear(66, 264)
        #self.bn1 = torch.nn.BatchNorm1d(hc)
      
    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device):  
        edge_weight00 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight00)
        x = F.relu(x)
        #print(x.shape)
        x11 = x
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        batch11 = batch

        #x11 = self.conv1(x11, edge_index11, batch11, edge_attr = edge_weight11)
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        #print(weights_sz.shape)
        #print(edge_index11.shape)
        x11 = F.relu(x11)
        x111 =  self.flatten(x11, batch)
        x111 = x111.to(device)  
        x111 = F.dropout(x111, p=0.5, training=self.training)
        x111 = self.lin111(F.relu(x111))
        x111 = self.lin1111(F.relu(x111))

        x22 = x
        edge_index22 = edge_index
        batch22 = batch
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight22 = edge_weight
        #x22 = self.conv2(x22, edge_index22, batch22, edge_attr = edge_weight22)
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        x22 = F.relu(x22)
        x222 =  self.flatten(x22, batch)
        x222 = x222.to(device)  
        x222 = F.dropout(x222, p=0.5, training=self.training)
        x222 = self.lin222(F.relu(x222))
        x222 = self.lin2222(F.relu(x222))
        
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all = torch.cat((x11, x22),1) #2640*128
        all = torch.reshape(all, ((len(torch.unique(batch))), 1, seg, all.shape[1])) ## can be really reshaped?
        all = all.to(device) #10*1*264*128
        all = self.conv4(all) #10*1*264*1
        all = torch.reshape(all, ((len(torch.unique(batch))), seg)) #10*264
        att = self.lin4(all)
        att = F.relu(att)
        att = self.lin5(att)  #10*264
        #att = F.relu(att)
        #att = self.softmax_func(att)
        att = torch.sigmoid(att)
        #att = torch.relu(att)
        att = torch.reshape(att, ((len(torch.unique(batch)))*seg, 1)) #(10*264)*1
        hidden_channels = x11.shape[1]
        attt = torch.tile(att, (1, hidden_channels))
        

        all1 = torch.mul(attt, x11)
        all2 = torch.mul(attt, x22)
        #all1 = self.flatten(all1, batch)
        #all2 = self.flatten(all2, batch)
        all1 = all1.to(device)
        all2 = all2.to(device)

        #print(all1.shape)
        
        #all = torch.reshape(all,(all.shape[0]*all.shape[1], 1))
        

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x11
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch

        x3 = x11
        edge_index3 = edge_index
        edge_weight3 = edge_weight
        batch3 = batch

        x4 = x11
        edge_index4 = edge_index
        edge_weight4 = edge_weight
        batch4 = batch

        x5 = x22
        edge_index5 = edge_index
        edge_weight5 = edge_weight
        batch5 = batch

        x6 = x22
        edge_index6 = edge_index
        edge_weight6 = edge_weight
        batch6 = batch

        x7 = x22
        edge_index7 = edge_index
        edge_weight7 = edge_weight
        batch7 = batch

        x8 = x22
        edge_index8 = edge_index
        edge_weight8 = edge_weight
        batch8 = batch

        #x1 = torch.cat((x1, all), 1)
        
        x1 = x1 + all2        
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)
        x1 = self.flatten(x1, batch1)
        x1 = x1.to(device)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(F.relu(x1))
        x1 = self.lin3(F.relu(x1))
        #print(x1)

        x2 = x2 + all2
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        x2 = self.flatten(x2, batch2)
        x2 = x2.to(device)  
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin12(F.relu(x2))
        x2 = self.lin32(F.relu(x2))
         

        x3 = x3 + all2
        x3, edge_index3, edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index3,edge_weight3, batch3)
        x3 = self.flatten(x3, batch3)
        x3 = x3.to(device)  
        x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.lin13(F.relu(x3))
        x3 = self.lin33(F.relu(x3))

        x4 = x4 + all2
        x4, edge_index4, edge_attr4, batch4, perm4, score4 = self.pool4(x4, edge_index4, edge_weight4, batch4)
        x4 = self.flatten(x4, batch4)
        x4 = x4.to(device)  
        #x4 = torch.cat((x4, all2),1)
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x4 = self.lin14(F.relu(x4))
        x4 = self.lin34(F.relu(x4))

        x5 = x5 + all1
        x5, edge_index5, edge_attr5, batch5, perm5, score5 = self.pool5(x5, edge_index5, edge_weight5, batch5)
        x5 = self.flatten(x5, batch5)
        x5 = x5.to(device) 
        #x5 = torch.cat((x5, all1),1) 
        x5 = F.dropout(x5, p=0.5, training=self.training)
        x5 = self.lin15(F.relu(x5))
        x5 = self.lin35(F.relu(x5))
        
        x6 = x6 + all1
        x6, edge_index6, edge_attr6, batch6, perm6, score6 = self.pool6(x6, edge_index6, edge_weight6, batch6)
        x6 = self.flatten(x6, batch6)
        x6 = x6.to(device)  
        #x6 = torch.cat((x6,all1),1) 
        x6 = F.dropout(x6, p=0.5, training=self.training)
        x6 = self.lin16(F.relu(x6))
        x6 = self.lin36(F.relu(x6))

        x7 = x7 + all1
        x7, edge_index7, edge_attr7, batch7, perm7, score7 = self.pool7(x7, edge_index7, edge_weight7, batch7)
        x7 = self.flatten(x7, batch7)
        x7 = x7.to(device)  
        #x7 = torch.cat((x7, all1),1)
        x7 = F.dropout(x7, p=0.5, training=self.training)
        x7 = self.lin17(F.relu(x7))
        x7 = self.lin37(F.relu(x7))

        x8 = x8 + all1
        x8, edge_index8, edge_attr8, batch8, perm8, score8 = self.pool8(x8, edge_index8, edge_weight8, batch8)
        x8 = self.flatten(x8, batch8)
        x8 = x8.to(device)   
        #x8 = torch.cat((x8, all1),1)
        x8 = F.dropout(x8, p=0.5, training=self.training)
        x8 = self.lin18(F.relu(x8))
        x8 = self.lin38(F.relu(x8))
       
        return x1, perm1, torch.sigmoid(score1), x2, perm2, torch.sigmoid(score2), x3, perm3, torch.sigmoid(score3), \
        x4, perm4, torch.sigmoid(score4), x5, perm5, torch.sigmoid(score5), x6, perm6, torch.sigmoid(score6), x7, perm7, \
        torch.sigmoid(score7), x8, perm8, torch.sigmoid(score8), att, x111, x222   
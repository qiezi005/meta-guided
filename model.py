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
from torch.nn import Linear, BatchNorm1d, Conv2d, Sequential, ReLU, Softmax, MultiheadAttention,Embedding
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GPSConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling, SAGPooling
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import remove_self_loops


class GPS_prompt_mt_2roi(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, pt_dim, pt_dim2, roi_pt_dim, sub_pt_dim, pe_dim):
        #print(sub_pt_dim, node_pt_dim)
        super(GPS_prompt_mt_2roi, self).__init__()
        roi_num = 264
        torch.manual_seed(12345)
        #self.lin1 = Linear(hidden_channels*116 + pt_dim2, hc2)
        ratio1 = 0.3 #0.9
        ratio2 = 0.3 #0.9
        hidden_channels = hidden_channels #+ node_pt_dim

        self.node_emb = Linear(hidden_channels, hidden_channels - pe_dim)
        self.pe_lin = Linear(roi_pt_dim, pe_dim)
        self.pe_norm = BatchNorm1d(roi_pt_dim)
        #self.edge_emb = Linear(4, hidden_channels)


        nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
        self.conv0 = GPSConv(hidden_channels, GINEConv(nn, edge_dim = 1), heads=4)

        #hidden_channels = hidden_channels + pt_dim
        
        self.node_emb1 = Linear(hidden_channels, hidden_channels - pt_dim)
        self.pe_lin1 = Linear(roi_pt_dim, pt_dim)
        self.pe_norm1 = BatchNorm1d(roi_pt_dim)

        self.node_emb2 = Linear(hidden_channels, hidden_channels - pt_dim)
        self.pe_lin2 = Linear(roi_pt_dim, pt_dim)
        self.pe_norm2 = BatchNorm1d(roi_pt_dim)


        self.conv1 = GATConv(hidden_channels, hidden_channels2)
        self.conv2 = GATConv(hidden_channels, hidden_channels2)
        self.softmax_func=Softmax(dim=1)
        self.multihead_attn = MultiheadAttention(hidden_channels2, 4,  batch_first='True')
        self.multihead_attn2 = MultiheadAttention(hidden_channels2, 4,  batch_first='True')

        self.lin11 = Linear(math.ceil(ratio1*roi_num)*hidden_channels2, hc2)
        self.lin12 = Linear(sub_pt_dim, hc2)
        self.lin13 = Linear(hc2*2, hc2)
        self.lin14 = Linear(hc2, 1)
        self.lin15 = Linear(hc2, 1)

        self.lin21 = Linear(math.ceil(ratio2*roi_num)*hidden_channels2, hc2)
        self.lin22 = Linear(sub_pt_dim, hc2)
        self.lin23 = Linear(hc2*2, hc2)
        self.lin24 = Linear(hc2, 2)
        self.lin25 = Linear(hc2, 2)

        self.lin4 = Linear(roi_pt_dim, pt_dim) #2048 llm2vec
        self.lin44 = Linear(roi_pt_dim, pt_dim) #2048 llm2vec
        #self.lin444 = Linear(roi_pt_dim, node_pt_dim) #2048 llm2vec
        self.lin5 = Linear(sub_pt_dim, pt_dim2) #2048 llm2vec
        self.lin6 = Linear(hc2, 2) #2048 llm2vec

        self.pool1 = TopKPooling(hidden_channels2, ratio=ratio1, multiplier=1, nonlinearity=torch.sigmoid)
        self.pool2 = TopKPooling(hidden_channels2, ratio=ratio2, multiplier=1, nonlinearity=torch.sigmoid)
    
        #self.double()
    def flatten(self, x, batch, device):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty((len(torch.unique(batch)),seg*x.shape[1]), device =device)
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x

    def forward(self, x, edge_index, edge_weight, batch, device, roi_pt_dim, sub_pt_dim, node_pt_dim):
        roi_num = 264
        
        seg = int(edge_weight.shape[0]/len(torch.unique(batch)))
        edge_weight_new = torch.empty(len(torch.unique(batch))*(seg-sub_pt_dim), device =device)
        sub_pt = torch.empty((len(torch.unique(batch)), sub_pt_dim), device =device)
        for i in range(0, len(torch.unique(batch))):
            edge_weight_new[i*(seg-sub_pt_dim):(i+1)*(seg-sub_pt_dim)] = edge_weight[i*seg:(i+1)*seg-sub_pt_dim] 
            sub_pt[i] = edge_weight[(i+1)*seg-sub_pt_dim:(i+1)*seg] 

        edge_weight = edge_weight_new
        #sub_pt = self.lin5(sub_pt)
        #sub_pt = F.leaky_relu(sub_pst, negative_slope=0.33)
        
        pro_emb = x[:, roi_num:roi_num+roi_pt_dim]
        pro_emb2 = x[:, roi_num+roi_pt_dim:roi_num+2*roi_pt_dim]
        pro_emb3 = x[:, roi_num+2*roi_pt_dim:]
    
        #print(pro_emb.shape)
        #print(pro_emb2.shape)
        x = x[:, :roi_num]
        seg = int(pro_emb.shape[0]/len(torch.unique(batch)))
        pro_x = torch.empty((len(torch.unique(batch)),seg, pro_emb.shape[1]), device =device)
        pro_x2 = torch.empty((len(torch.unique(batch)),seg, pro_emb.shape[1]), device =device)
        pro_x3 = torch.empty((len(torch.unique(batch)),seg, pro_emb.shape[1]), device =device)
        for i in range(0, len(torch.unique(batch))):
            pro_x[i] = pro_emb[i*seg:(i+1)*seg]  ##cog
            pro_x2[i] = pro_emb2[i*seg:(i+1)*seg] ##sz
            pro_x3[i] = pro_emb3[i*seg:(i+1)*seg] ##sz
        #pro_x = self.lin4(pro_x)
        #pro_x2 = self.lin44(pro_x2)
        #pro_x3 = self.lin444(pro_x3)
        pro_xx = torch.empty((len(torch.unique(batch))*seg, pro_x.shape[2]), device =device)
        pro_xx2 = torch.empty((len(torch.unique(batch))*seg, pro_x2.shape[2]), device =device)
        pro_xx3 = torch.empty((len(torch.unique(batch))*seg, pro_x3.shape[2]), device =device)
        for i in range(0, len(torch.unique(batch))):
            pro_xx[i*seg:(i+1)*seg] = pro_x[i] 
            pro_xx2[i*seg:(i+1)*seg] = pro_x2[i] 
            pro_xx3[i*seg:(i+1)*seg] = pro_x3[i] 
        #print(pro_xx.shape)
        #sub_pt3 = self.flatten(pro_xx3, batch, device)
        #sub_pt3 = self.lin31(sub_pt3)
        #sub_pt3 = self.softmax_func(sub_pt3)
        #x = x + pro_xx3

        #print(pro_xx3.shape)
        
        x_pe = self.pe_norm(pro_xx3)
        x = torch.cat((self.node_emb(x), torch.sigmoid(self.pe_lin(x_pe))), 1)
        #print(self.node_emb(x).shape)
        #x = torch.cat([x, x_pe], dim = 1)
        edge_weight = torch.reshape(edge_weight,(len(edge_weight), 1))
        #edge_attr = self.edge_emb(edge_weight)
        x = self.conv0(x, edge_index, batch, edge_attr = edge_weight)
        
        #x = F.leaky_relu(x, negative_slope=0.33)
        x = F.relu(x)

        x_pe1 = self.pe_norm1(pro_xx)
        x11 = torch.cat((self.node_emb1(x), torch.sigmoid((self.pe_lin1(x_pe1)))), 1)
        #x11 = torch.cat([x, pro_xx], dim = 1)
        edge_index11 = edge_index
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        edge_weight11 = edge_weight
        #edge_weight11 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x11 = self.conv1(x11, edge_index11, edge_weight11)
        #x11 = self.conv1(x11, edge_index11, batch, edge_attr = edge_weight11)
        x11 = F.relu(x11)

        x_pe2 = self.pe_norm2(pro_xx2)
        x22 = torch.cat((self.node_emb2(x), torch.sigmoid(self.pe_lin2(x_pe2))), 1)
        #x22 = torch.cat([x, pro_xx2], dim = 1)
        edge_index22 = edge_index
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1)) #edge_index
        edge_weight22 = edge_weight
        #edge_weight22 = torch.reshape(edge_weight,(len(edge_weight), 1))
        x22 = self.conv2(x22, edge_index22, edge_weight22)
        #x22 = self.conv2(x22, edge_index22, batch, edge_attr = edge_weight22)
        x22 = F.relu(x22)
        
        seg = int(x22.shape[0]/len(torch.unique(batch)))
        all11 = torch.reshape(x11, ((len(torch.unique(batch))), seg, x11.shape[1])) ## can be really reshaped?
        #print(all11.shape)
        all22 = torch.reshape(x22, ((len(torch.unique(batch))), seg, x22.shape[1]))

        all1, att = self.multihead_attn(all11, all22, all22)
        all2, att2 = self.multihead_attn2(all22, all11, all11)
        all1 = torch.reshape(all1, ((len(torch.unique(batch)))*seg, all1.shape[2]))
        all2 = torch.reshape(all2, ((len(torch.unique(batch)))*seg, all2.shape[2]))

        x1 = x11
        edge_index1 = edge_index
        edge_weight1 = edge_weight
        batch1 = batch

        x2 = x22
        edge_index2 = edge_index
        edge_weight2 = edge_weight
        batch2 = batch

        x1 = all1 + x1
        x1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index1,edge_weight1, batch1)  
        x1 = self.flatten(x1, batch1, device)
        x1 = self.lin11(x1)
        x1 = F.dropout(F.relu(x1), p=0.2, training=self.training)
        sub_pt1 = self.lin12(sub_pt)
        sub_pt1 = F.dropout(F.relu(sub_pt1), p=0.2, training=self.training)
        x_out1 = x1
        sub_out1 = sub_pt1
        x1 = torch.cat([x1, sub_pt1], dim = 1)
        #x = x+sub_pt
        x1 = self.lin13(x1)
        #x = F.leaky_relu(x, negative_slope=0.33) 
        x1 = F.relu(x1)
        x1 = self.lin14(x1)
        sub_pt1 = self.lin15(sub_pt1)

        x2 = all1 + x2
        x2, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index2,edge_weight2, batch2)
        #x2 = torch.cat((x2, all1), dim=0)
        x2 = self.flatten(x2, batch2, device)
        x2 = self.lin21(x2)
        x2 = F.dropout(F.relu(x2), p=0.2, training=self.training)
        sub_pt2 = self.lin12(sub_pt)
        sub_pt2 = F.dropout(F.relu(sub_pt2), p=0.2, training=self.training)
        x_out2 = x2
        sub_out2 = sub_pt2
        
        x2 = torch.cat([x2, sub_pt2], dim = 1)
        #x = x+sub_pt
        x2 = self.lin23(x2)
        #x = F.leaky_relu(x, negative_slope=0.33) 
        x2 = F.relu(x2)
        x2 = self.lin24(x2)
        #x2 = self.softmax_func(x2)
      
        sub_pt2 = self.lin25(sub_pt2)
       # sub_pt2 = self.softmax_func(sub_pt2)
        
        return x1, sub_pt1, x2, sub_pt2, x_out1, sub_out1, x_out2, sub_out2, perm1, torch.sigmoid(score1), perm2, torch.sigmoid(score2), att

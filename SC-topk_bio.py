"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import math
from utils import get_data
from utils import create_dataset
from torch_geometric.data import DataLoader
from model import MT_GAT_topk_shareEn_multiple8_joint_last, MT_GAT_topk_shareEn_multiple8_joint_cross
import torch.nn
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torchmetrics.regression import KendallRankCorrCoef

#os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MAELoss(yhat,y):
    return torch.mean(torch.abs(yhat-y))
def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def L1Loss(model, alpha):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + (0.5*alpha * torch.sum(torch.abs(parma)))
    return l1_loss

def write_attention(filename, train_attention, train_node_index, fold, task):
    #train_attention = np.float32(train_attention.cpu().detach().numpy())
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy" 
    np.save(filename1, train_node_index)
    mean_train = np.mean(train_node_index, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

def write_attention2(filename, train_attention, fold, task):
    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_attention)
    mean_train = np.mean(train_attention, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

def write_attention3(filename, train_index_szf, train_weight_szf, fold, task):

    filename1 = filename + '/train_attention' + str(fold) + task + ".npy" 
    np.save(filename1, train_weight_szf)
    mean_train = np.mean(train_weight_szf, axis=0)
    filename2 = filename + '/train_mean_attention' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 

    filename1 = filename + '/train_node_index' + str(fold) + task + ".npy" 
    np.save(filename1, train_index_szf)
    mean_train = np.mean(train_index_szf, axis=0)
    filename2 = filename + '/train_mean_node_index' + str(fold) + task + ".npy"
    np.save(filename2, mean_train) 


def train(model_to_train, train_dataset_loader, model_optimizer,test_score, test_score2, weight, sz_mark, cog_mark, rr1, rr2, device):
    model_to_train.train()
    

    for data in train_dataset_loader:  # Iterate in batches over the training dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        out, index, att, out2, index2, att2, att9  = model_to_train(data.x.float(), data.edge_index, data.edge_weight, data.batch, device)  # Perform a single forward pass.
        #, x111, x222, edge_sz, weight_sz, edge_cog, weight_cog
        
        test_tem = test_score[data.y]
        test_tem2 = test_score2[data.y]
        #for i in range(0, len(torch.unique(batch))):
        #print(torch.reshape(index,(1,index.shape[0])))
        roinum = 264 
        #rr1 = 0.4 ###############################################remember modify################
        #rr2 = 0.2 ###############################################remember modify################
        ratio1 = math.ceil(rr1*264)
        ratio2 = math.ceil(rr2*264)
        index = index%roinum#torch.reshape(index%roinum,(1,index.shape[0]))
        index2 = index2%roinum#torch.reshape(index2%roinum,(1,index2.shape[0]))
        
        corr_loss1 = 0
        diff = 0
        dif1 = 0
        dif2 = 0
        attt1 = torch.zeros(roinum, device = device)
        attt2 = torch.zeros(roinum, device = device)
        for i in range(0, len(torch.unique(data.batch))):
            sub_att = torch.zeros(roinum,2, device = device)
            sub_att[index[i*ratio1:(i+1)*ratio1],0] = att[i*ratio1:(i+1)*ratio1]
            sub_att[index2[i*ratio2:(i+1)*ratio2],1] = att2[i*ratio2:(i+1)*ratio2]
            #print(sub_att.shape)
            sub_att2 = torch.transpose(sub_att, 0, 1)
         #   print(sub_att.shape)
            #corr_loss = torch.corrcoef(sub_att2)
            #corr_loss1 = corr_loss1 + torch.sum(torch.abs(corr_loss[0, 1]-0.3))
            ##ratio11 = 26
            ##ratio22 = 26
            index_cog = torch.topk(cog_mark[:,0], ratio1).indices
            index_sz = torch.topk(sz_mark[:,0], ratio2).indices
            ##index_att1 = torch.topk(sub_att[:,0],ratio11).indices
            ##index_att2 = torch.topk(sub_att[:,1],ratio22).indices

            ##over1 = np.intersect1d(index_att1.cpu().detach().numpy(),index_cog.cpu().detach().numpy())
            ##over2 = np.intersect1d(index_att2.cpu().detach().numpy(),index_sz.cpu().detach().numpy())
            #print(over1.shape[0])
            #print(over2.shape[0])
            ##dif1 = torch.abs(1-torch.tensor(over1.shape[0]/ratio11)) + dif1
            ##dif2 = torch.abs(1-torch.tensor(over2.shape[0]/ratio22)) + dif2


            cog_mark2 = torch.zeros(roinum,2, device = device)
            sz_mark2 = torch.zeros(roinum,2, device = device)
            cog_mark2[index_cog,:] = cog_mark[index_cog,:]
            sz_mark2[index_sz,:] = sz_mark[index_sz,:]
            
##########################################sum bio_loss##############################################################
            #diff1 = torch.abs(sub_att[:,0] - cog_mark2[:,0])
            #diff2 = torch.abs(sub_att[:,1] - sz_mark2[:,0])
            #diff = torch.sum(diff1) + torch.sum(diff2) + diff
##########################################pearson bio_loss##############################################################
            #print(attt1.shape)
            #print(sub_att[:,0].shape)
            #attt1 = sub_att[:,0] + attt1
            #print(attt1.shape)
            #attt2 = sub_att[:,1] + attt2
            diff1 = torch.concat([torch.reshape(sub_att[:,0], (1,roinum)), torch.reshape(cog_mark2[:,0], (1,roinum))],dim=0)
            diff2 = torch.concat([torch.reshape(sub_att[:,1], (1,roinum)), torch.reshape(sz_mark2[:,0], (1,roinum))],dim=0)
            diff11 = torch.corrcoef(diff1)
            diff22 = torch.corrcoef(diff2)
           # print(diff11[0, 1], diff11[1, 1])
            ##kendall = KendallRankCorrCoef()
            #print(torch.reshape(sub_att[:,0], (1,roinum)).shape)
            #print(torch.reshape(cog_mark2[:,0], (1,roinum)).shape)
            ##diff1 = kendall(sub_att[:,0], cog_mark2[:,0])
            ##diff2 = kendall(sub_att[:,0], sz_mark2[:,0])
            #diff1 = kendall(torch.reshape(sub_att[:,0], (1,roinum)), torch.reshape(cog_mark2[:,0], (1,roinum)))
            #diff2 = kendall(torch.reshape(sub_att[:,1], (1,roinum)), torch.reshape(sz_mark2[:,0], (1,roinum)))
            dif1 = dif1 + torch.abs(1-diff11[0, 1])#1[0, 1]) #torch.abs(1-diff11[0, 1])#torch.abs(diff11[0, 1]))
            dif2 = dif2 + torch.abs(1-diff22[0, 1])#2[0, 1])#torch.abs(diff22[0, 1]))
            #diff = diff11[0, 1] + diff22[0, 1] + diff
            #corr_loss2 = corr_loss2 + torch.sum(torch.abs(corr_loss[0:4, 0:4]-0.9))
            #corr_loss3 = corr_loss3 + torch.sum(torch.abs(corr_loss[4:8, 4:8]-0.9))
        #print(diff/len(torch.unique(data.batch))/264)
        #bio_loss = diff/len(torch.unique(data.batch))
        #attt1 = attt1/len(torch.unique(data.batch))
        #attt2 = attt2/len(torch.unique(data.batch))
        #diff1 = torch.concat([torch.reshape(attt1, (1,roinum)), torch.reshape(cog_mark2[:,0], (1,roinum))],dim=0)
        #diff2 = torch.concat([torch.reshape(attt2, (1,roinum)), torch.reshape(sz_mark2[:,0], (1,roinum))],dim=0)
        #diff11 = torch.corrcoef(diff1)
        #diff22 = torch.corrcoef(diff2)
        #sz_loss = diff22[0, 1]
        #cog_loss = diff11[0, 1]
        sz_loss = dif2/len(torch.unique(data.batch))
        cog_loss = dif1/len(torch.unique(data.batch))
       # bio_loss = diff/len(torch.unique(data.batch))/ratio
        reg_loss = RMSELoss(out, torch.reshape(test_tem.float(),out.shape))
        class_loss = F.cross_entropy(out2, test_tem2)
        loss = reg_loss + class_loss + L2Loss(model_to_train, 0.001) + weight*sz_loss + weight*cog_loss#((1-sz_loss) + (1-cog_loss)) #+ corr_loss1 
        #print(sz_loss)
        #print(cog_loss)
        #print(class_loss)
        #print(reg_loss, class_loss, bio_loss)
        #print(-corr_loss1*2, ac_loss, loss)
        if loss == 'nan':
            break
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.

def test(model, loader, test_score, test_score2, device):
   
    out_sum = torch.tensor(()).to(device)
    true_sum = torch.tensor(()).to(device)
    attention = torch.tensor(()).to(device)
    node_index = torch.tensor(()).to(device)

    out_sum2 = torch.tensor(()).to(device)
    true_sum2 = torch.tensor(()).to(device)
    attention2 = torch.tensor(()).to(device)
    node_index2 = torch.tensor(()).to(device)
    attention9 = torch.tensor(()).to(device)

    model.eval()
    with torch.no_grad():
    
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.x, data.edge_index, data.edge_weight,data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
            test_tem = test_score[data.y]
            test_tem2 = test_score2[data.y]
            
            out, index, att, out2, index2, att2, att9 = model(data.x.float(), data.edge_index, data.edge_weight, data.batch, device) 
            # , x111, x222, edge_sz, weight_sz, edge_cog, weight_cog
            # print(out_sum.shape)
            out_sum = torch.cat((out_sum, out), dim=0)       
            true_sum = torch.cat((true_sum, torch.reshape(test_tem.float(), out.shape)), dim=0)
            node_index = torch.cat((node_index, index), dim=0)
            attention = torch.cat((attention, att), dim=0)

            out2 = torch.argmax(out2, dim = 1)
            out_sum2 = torch.cat((out_sum2, out2), dim=0)       
            true_sum2 = torch.cat((true_sum2, torch.reshape(test_tem2.float(), out2.shape)), dim=0)
            node_index2 = torch.cat((node_index2, index2), dim=0)
            attention2 = torch.cat((attention2, att2), dim=0)
            attention9 = torch.cat((attention9, att9), dim=0)
        rmse = RMSELoss(out_sum, torch.reshape(true_sum.float(),out_sum.shape))
        mae = MAELoss(out_sum, torch.reshape(true_sum.float(),out_sum.shape))

        truev = torch.reshape(true_sum,(len(out_sum),1))
        out = np.squeeze(np.asarray(out_sum.cpu().detach().numpy()))
        truev = np.squeeze(np.asarray(truev.cpu().detach().numpy()))       
        corr = scipy.stats.pearsonr(out, truev)
        
        truev2 = torch.reshape(true_sum2,(len(out_sum2),1))
        out2 = np.squeeze(np.asarray(out_sum2.cpu().detach().numpy()))
        truev2 = np.squeeze(np.asarray(truev2.cpu().detach().numpy()))       
        corr2 =  np.sum(out2 == truev2)/truev.shape[0]

        attention = np.squeeze(np.asarray(attention.cpu().detach().numpy()))
        node_index = np.squeeze(np.asarray(node_index.cpu().detach().numpy()))

        attention2 = np.squeeze(np.asarray(attention2.cpu().detach().numpy()))
        node_index2 = np.squeeze(np.asarray(node_index2.cpu().detach().numpy()))

        attention9 = np.squeeze(np.asarray(attention9.cpu().detach().numpy()))
     

    return corr[0], rmse, mae, out, truev, node_index, attention, corr2, out2, truev2, node_index2, attention2, attention9
           
def transform_inver(target_scaler, test_out, test_true,train_out, train_true):

    test_outt = target_scaler.inverse_transform(test_out.reshape(-1,1))[:,0]    
    test_truet = target_scaler.inverse_transform(test_true.reshape(-1,1))[:,0]
    train_outt = target_scaler.inverse_transform(train_out.reshape(-1,1))[:,0]    
    train_truet = target_scaler.inverse_transform(train_true.reshape(-1,1))[:,0]
    return test_outt, test_truet, train_outt, train_truet

def transform(train_scoreo, test_scoreo):
    target_scaler = preprocessing.StandardScaler().fit(train_scoreo.reshape(-1,1))
    train_score = target_scaler.transform(train_scoreo.reshape(-1,1))[:,0] 
    test_score = target_scaler.transform(test_scoreo.reshape(-1,1))[:,0]
    return target_scaler, train_score, test_score
def generate_train(all_score, X_train, X_test):
    
    train_scoreo = all_score[X_train]
    test_scoreo = all_score[X_test]
    return train_scoreo, test_scoreo

def compute_RSME(test_outt, test_truet, train_outt,train_truet):
    rmse_test = RMSELoss(torch.tensor(test_outt), torch.tensor(test_truet))
    MAE_test = MAELoss(torch.tensor(test_outt),torch.tensor(test_truet))
    rmse_train = RMSELoss(torch.tensor(train_outt), torch.tensor(train_truet))
    MAE_train = MAELoss(torch.tensor(train_outt),torch.tensor(train_truet))
    return rmse_test, MAE_test, rmse_train, MAE_train

def format_trans(train_corr, test_corr, test_rmse, test_mae):
    train_corr = torch.tensor(np.float32(train_corr))
    test_corr = torch.tensor(np.float32(test_corr))
    test_rmse = torch.tensor(np.float32(test_rmse.cpu().detach().numpy()))
    test_mae = torch.tensor(np.float32(test_mae.cpu().detach().numpy()))
    return train_corr, test_corr, test_rmse, test_mae

def thebest(test_corr, rmse_test, MAE_test, test_out, train_out, train_true, test_true, test_outt, train_outt, train_truet, test_truet, train_attention, train_node_index):
    test_corrf = test_corr
    test_rmsef = rmse_test
    test_maef = MAE_test

    test_outf = test_out
    train_outf = train_out
    train_truef = train_true
    test_truef = test_true

    test_outtf = test_outt
    train_outtf = train_outt
    train_truetf = train_truet
    test_truetf = test_truet

    train_attentionf = train_attention
    train_node_indexf = train_node_index
    return test_corrf, test_rmsef, test_maef, test_outf, train_outf, train_truef, test_truef, test_outtf, train_outtf, train_truetf, test_truetf, train_attentionf, train_node_indexf 

def thebest2(test_corr,test_out, train_out, train_true, test_true, train_attention, train_node_index):
    test_corrf = test_corr
    test_outf = test_out
    train_outf = train_out
    train_truef = train_true
    test_truef = test_true

    train_attentionf = train_attention
    train_node_indexf = train_node_index
    return test_corrf, test_outf, train_outf, train_truef, test_truef, train_attentionf, train_node_indexf 


def app(test_corr_sf, test_rmse_sf, test_mae_sf, test_corr_s, test_rmse_s, test_mae_s, test_corrf, test_rmsef, test_maef, test_corr, test_rmse, test_mae):
    test_corr_sf.append(test_corrf)
    test_rmse_sf.append(test_rmsef)
    test_mae_sf.append(test_maef)
    test_corr_s.append(test_corr)
    test_rmse_s.append(test_rmse)
    test_mae_s.append(test_mae)
    return test_corr_sf, test_rmse_sf, test_mae_sf, test_corr_s, test_rmse_s, test_mae_s

def app2(test_corr_sf, test_corr_s, test_corrf, test_corr):
    test_corr_sf.append(test_corrf)
    test_corr_s.append(test_corr)
   
    return test_corr_sf,  test_corr_s

def compute_corr(test_corr_s, test_rmse_s, test_mae_s):
    corr = torch.tensor(np.mean(np.array(test_corr_s,dtype=np.float32)))
    rmse = torch.tensor(np.mean(np.array(test_rmse_s,dtype=np.float32)))
    mae = torch.tensor(np.mean(np.array(test_mae_s,dtype=np.float32)))
    return corr, rmse, mae  

hidden_channels=64
hidden_channels2=64
hc = 128 #12672 #8448 #3712  #3712 #1120 #1856 #3712
hc2= 128
weight = 1 #0.5
ratio1=0.1    ###number of node 60
ratio2=0.1          #256
epoch_num = 40
decay_rate = 0.005
decay_step = 10
lr = 0.001#0.001
num_folds = 5
batch_size = 10
runnum = 'e10'
kk = 264
print("\n---------Starting to load Data---------\n")
task='positive'
task2='cognition'
task5= 'PS'
task6= 'BACS'
task7= 'WM'
task8= 'VL'
connpath = '/home2/jing001/Multi-task-cons/COBRE/X.npy'  ##need to modify
scorepath = '/home2/jing001/Multi-task-cons/COBRE/Y_' + task +'.npy'  ##need to modify
scorepath5 = '/home2/jing001/Multi-task-cons/COBRE/Y_' + task5 +'.npy'  ##need to modify
scorepath6 = '/home2/jing001/Multi-task-cons/COBRE/Y_' + task6 +'.npy'  ##need to modify
scorepath7 = '/home2/jing001/Multi-task-cons/COBRE/Y_' + task7 +'.npy'  ##need to modify
scorepath8 = '/home2/jing001/Multi-task-cons/COBRE/Y_' + task8 +'.npy'  ##need to modify

timefile = '/home2/jing001/Multi-task-cons/MT-SZ/results_FC/COBRE_bio_pear_cross_'  +str(hidden_channels) + '_' + str(lr) + '_' + \
str(epoch_num) + '_' + str(hc2) + '_' + str(runnum) + '_' +  str(ratio1) +  str(ratio2)+ '_' + str(weight) + \
'_'+ str(kk) + '_'+str(int(time.time()))

sz_mark = np.load('/home2/jing001/Multi-task-cons/MT-SZ/sz_marker.npy')
cog_mark = np.load('/home2/jing001/Multi-task-cons/MT-SZ/cog_marker.npy')


os.mkdir(timefile)
               
all_data, all_score = get_data(connpath, scorepath)
all_data, all_score5 = get_data(connpath, scorepath5)
all_data, all_score6 = get_data(connpath, scorepath6)
all_data, all_score7 = get_data(connpath, scorepath7)
all_data, all_score8 = get_data(connpath, scorepath8)
all_score2 = all_score
all_score2[all_score2>0] = 1
all_score = (all_score5 + all_score6 + all_score7 + all_score8)/4

print(np.max(all_score))
print(np.min(all_score))

for i in range(len(all_data)):
    feature_matrix_ori = np.array(all_data[i])

num = all_data.shape[0]
#kf = KFold(n_splits=num_folds, shuffle=True)
stratifiedKFolds = StratifiedKFold(n_splits = num_folds, shuffle = True)

print("\n--------Split and Data loaded-----------\n")
fold = 0
true_out = np.squeeze(np.array([[]]))
pred_out = np.squeeze(np.array([[]]))

true_out2 = np.squeeze(np.array([[]]))
pred_out2 = np.squeeze(np.array([[]]))

test_corr_s = []
test_rmse_s = []
test_mae_s = []

test_corr_s2 = []
test_rmse_s2 = []
test_mae_s2 = []

test_corr_sf = []
test_rmse_sf = []
test_mae_sf = []

test_corr_sf2 = []
test_rmse_sf2 = []
test_mae_sf2 = []

epoch_sf = []
for X_train, X_test in stratifiedKFolds.split(all_data, all_score2):#kf.split(list(range(1,num))):
    print(fold)
    fold = fold+1
    
    model = MT_GAT_topk_shareEn_multiple8_joint_cross(hidden_channels, hidden_channels2, hc, hc2, ratio1, ratio2)
    #print("Model:\n\t",model)
    print(torch.cuda.is_available())
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    sz_mark = torch.tensor(sz_mark, dtype = torch.float).to(device)
    cog_mark = torch.tensor(cog_mark, dtype = torch.float).to(device)
    fine_file = '/home2/jing001/Multi-task-cons/MT-SZ/'+runnum + '/' 
    train_file = fine_file + 'Train' +str(fold) +'.txt'
    test_file = fine_file + 'Test' +str(fold) +'.txt'
    X_train = []
    X_test = []
    with open(train_file) as f:
        for line in f.readlines():
            X_train.append(line.strip('\n'))
    X_train = [eval(i) for i in X_train]    
       
    with open(test_file) as f:
        for line in f.readlines():
            X_test.append(line.strip('\n'))
    X_test = [eval(i) for i in X_test]

    train_data = all_data[X_train]
    test_data = all_data[X_test]

    train_scoreo, test_scoreo = generate_train(all_score, X_train, X_test)    
    train_score2, test_score2 = generate_train(all_score2, X_train, X_test)  
    
    target_scaler, train_score, test_score = transform(train_scoreo, test_scoreo)
    
    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))

    training_dataset = create_dataset(train_data, index_train, kk)
    testing_dataset = create_dataset(test_data, index_test, kk)

    train_score_input = torch.tensor(train_score).to(device)
    test_score_input = torch.tensor(test_score).to(device)

    train_score_input2 = torch.tensor(train_score2).to(device)
    test_score_input2 = torch.tensor(test_score2).to(device)


    with open('traindata.txt', 'w') as f:
        f.write(str(training_dataset))
    print(len(training_dataset))
    
    train_loader = DataLoader(training_dataset, batch_size, shuffle = True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(testing_dataset, batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay_rate)


    train_path = timefile+'/Train'+str(fold)+'.txt'
    test_path = timefile+'/Test'+str(fold)+'.txt'

    file=open(train_path,'w')  
    for line in X_train:
        line = str(line)
        # print(line)
        file.write(line+'\n')
    file.close()

    file=open(test_path,'w')  
    for line in X_test:
        line = str(line)
        # print(line)
        file.write(line+'\n')
    file.close()

    test_corrf = -10
    test_corrf2 = -10
    
    epochf = 0
    test_rmsef = 0
    test_maef = 0

    for epoch in range(1, epoch_num):
        #if epoch % decay_step == 0:
            #for p in optimizer.param_groups:
                #p['lr'] *= decay_rate
        train(model, train_loader, optimizer,train_score_input,train_score_input2, weight, sz_mark, cog_mark,ratio1, ratio2, device)
        
        train_corr, train_rmse, train_mae, train_out, train_true, train_node_index, train_attention, \
        train_corr2, train_out2, train_true2, train_node_index2, train_attention2, train_attention9 = \
        test(model, train_loader,train_score_input,train_score_input2, device)
        #, train_index_sz,train_weight_sz, train_index_cog, train_weight_cog
        test_corr, test_rmse, test_mae, test_out, test_true, test_node_index, test_attention, \
        test_corr2, test_out2, test_true2, test_node_index2, test_attention2,test_attention9 = \
        test(model, test_loader,test_score_input,test_score_input2, device)
        #, test_index_sz,test_weight_sz, test_index_cog, test_weight_cog
        #######################inverse
        test_outt, test_truet, train_outt, train_truet = transform_inver(target_scaler, test_out, test_true,train_out, train_true)
        
        rmse_test, MAE_test, rmse_train, MAE_train = compute_RSME(test_outt, test_truet, train_outt,train_truet)        
    
        ##########################################
        if epoch % 1 == 0:
            print(epoch)
            print(f'Epoch: {epoch:03d}, Train_corr: {train_corr:.4f}, Train_rmse: {train_rmse:.4f},Train_mae: {train_mae:.4f}')
            print(f'Epoch: {epoch:03d}, Test_corr: {test_corr:.4f}, Test_rmse: {test_rmse:.4f},Test_mae: {test_mae:.4f}')

            print(f'Epoch: {epoch:03d}, Train_accu: {train_corr2:.4f}')
            print(f'Epoch: {epoch:03d}, Test_accu: {test_corr2:.4f}')
            #print(train_corr.float(), train_corr.shape)
            train_corr, test_corr, test_rmse, test_mae = format_trans(train_corr, test_corr, test_rmse, test_mae)
            train_corr2 = torch.tensor(np.float32(train_corr2)) 

            if  test_corrf2 + test_corrf <  test_corr2 +test_corr:
                test_corrf, test_rmsef, test_maef, test_outf, train_outf, train_truef, test_truef, test_outtf, \
                train_outtf, train_truetf, test_truetf, train_attentionf, train_node_indexf = \
                thebest(test_corr, rmse_test, MAE_test, test_out, train_out, train_true, test_true, test_outt, train_outt, train_truet, test_truet, train_attention, train_node_index)
                test_corrf2,  test_outf2, train_outf2, train_truef2, test_truef2, train_attentionf2, train_node_indexf2 = \
                thebest2(test_corr2, test_out2, train_out2, train_true2, test_true2, train_attention2, train_node_index2)
                train_attentionf9 = train_attention9
                test_attentionf9 = test_attention9
                epochf = epoch
                #save checkpoints
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, timefile)

    test_corr_sf, test_rmse_sf, test_mae_sf, test_corr_s, test_rmse_s, test_mae_s = \
    app(test_corr_sf, test_rmse_sf, test_mae_sf, test_corr_s, test_rmse_s, test_mae_s, test_corrf, test_rmsef, test_maef, test_corr, test_rmse, test_mae)
    test_corr_sf2, test_corr_s2 = app2(test_corr_sf2,  test_corr_s2, test_corrf2, test_corr2)
    
    epoch_sf.append(epochf)
    pred_out = np.concatenate((pred_out, test_outtf), axis=0)
    true_out = np.concatenate((true_out, test_truetf), axis=0)
    pred_out2 = np.concatenate((pred_out2, test_outf2), axis=0)
    true_out2 = np.concatenate((true_out2, test_truef2), axis=0)
    
    
corr, rmse, mae = compute_corr(test_corr_s, test_rmse_s, test_mae_s)
corr2 = torch.tensor(np.mean(np.array(test_corr_s2,dtype=np.float32)))



corrf, rmsef, maef = compute_corr(test_corr_sf, test_rmse_sf, test_mae_sf)
corrf2 = torch.tensor(np.mean(np.array(test_corr_sf2,dtype=np.float32)))


final2 = [corrf, rmsef, maef, corrf2]
print(corrf, rmsef, maef)
print(corrf2)
print(epoch_sf)


report_best = classification_report(pred_out2, true_out2)
print(report_best)
torch.cuda.empty_cache()



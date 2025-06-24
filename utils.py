import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from sklearn import preprocessing
import heapq

def get_data(connpath, scorepath, feature = False):
    all_data = np.load(connpath)
    all_score = np.load(scorepath)
    
    return all_data, all_score

# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset
def create_fusion_dataset(data, data2, node_FC_data, node_SC_data, score, features = None):
    dataset_list = []
    n = data.shape[1]
    kk = 22#22 17
    for i in range(len(data)):
       # degree_matrix = np.count_nonzero(data[i], axis=1).reshape(n,1)
       # weight_matrix = np.diag(np.sum(data[i], axis=1)).diagonal().reshape(n,1)
        #feature_matrix = np.hstack((degree_matrix, weight_matrix))
        feature_matrix_ori = np.array(data[i])
        feature_matrix_ori_SC = np.array(data2[i])
        node_FC = np.array(node_FC_data[i])
        node_SC = np.array(node_SC_data[i])

        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
        feature_matrix_ori_SC2 =  feature_matrix_ori_SC/np.max(feature_matrix_ori_SC)
        #print(node_FC.shape, node_SC.shape)
        #print(np.max(node_FC),np.min(node_FC))
        node_FC = node_FC/np.max(node_FC)
        feature_matrix = node_FC
       
        #print(np.max(node_SC),np.min(node_SC))
        node_SC = node_SC/np.max(node_SC)
        feature_matrix2 = node_SC
        feature_matrix_total = np.concatenate([feature_matrix, feature_matrix2], axis = 0)
        #print(feature_matrix_total.shape)

        edge_index_coo = np.triu_indices(116, k=1)
        edge_index_coo2 = np.triu_indices(116, k=1)
        edge_adj = np.zeros((116, 116))
        for ii in range(len(feature_matrix_ori2[1])):
            #index_max = np.argmax(feature_matrix_ori2[ii], n=kk)
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        #print(edge_adj)
        #edge_weight = feature_matrix_ori2[edge_index_coo]
        edge_weight = edge_adj[edge_index_coo]
        edge_weight2 = feature_matrix_ori_SC2[edge_index_coo2]
        edge_weight2[edge_weight2<0.6] = 0
        #edge_weight3 = np.reshape(np.ones(116,int),(1,116))
        edge_weight3 = np.ones(116,int)
        #edge_weight3 = edge_weight3 * 0.6
        #print(edge_weight.shape, edge_weight2.shape, edge_weight3.shape)
        edge_weight_total = np.concatenate([edge_weight, edge_weight2, edge_weight3], axis = 0)
        #print(edge_weight_total.shape)

        edge_index_coo = torch.tensor(edge_index_coo)
        edge_index_coo2 = torch.tensor(edge_index_coo2)
        edge_index_coo2 = edge_index_coo2+116
        edge_index_coo4=np.reshape(np.arange(0,116), (1,116))
        edge_index_coo5=np.reshape(np.arange(116,232), (1,116))
        edge_index_coo3 = torch.tensor(np.concatenate((edge_index_coo4,edge_index_coo5),axis=0))
        
        edge_index_coo_total = torch.cat((edge_index_coo, edge_index_coo2, edge_index_coo3), 1)
        #print(edge_index_coo_total.shape)
        

        #edge_weight = torch.tensor(edge_weight)
       #print(edge_weight)
       # print(edge_index_coo)
        graph_data = Data(x = torch.tensor(feature_matrix_total, dtype = torch.float32), edge_index=edge_index_coo_total, edge_weight=torch.tensor(edge_weight_total, dtype = torch.float32), node_FC = torch.tensor(node_FC, dtype = torch.float32), node_SC =torch.tensor(node_SC, dtype = torch.float32), y = torch.tensor(score[i]))
        #graph_data = Data(x = torch.tensor(feature_matrix2, dtype = torch.float32), edge_index=edge_index_coo2, edge_weight=torch.tensor(edge_weight2, dtype = torch.float32), y = torch.tensor(score[i]))
        
        #print(np.reshape(feature_matrix,(1,data.shape[1],data.shape[2])))
        dataset_list.append(graph_data)
    #print(torch.tensor(dataset_list).x)
    return dataset_list

def create_dataset(data, indexx, kk, features = None):
    dataset_list = []
    n = data.shape[0]
    kk = kk#22 17
    for i in range(len(data)):
        #print(i)
       # degree_matrix = np.count_nonzero(data[i], axis=1).reshape(n,1)
       # weight_matrix = np.diag(np.sum(data[i], axis=1)).diagonal().reshape(n,1)
        #feature_matrix = np.hstack((degree_matrix, weight_matrix))
        feature_matrix_ori = np.array(data[i])
        #print(feature_matrix_ori.shape)
        #target_scaler = preprocessing.StandardScaler().fit(feature_matrix_ori.reshape(-1,1))
        #feature_matrix_ori = target_scaler.transform(feature_matrix_ori.reshape(-1,1))[:,0]    
        #feature_matrix_ori = feature_matrix_ori.reshape(116,116)
        #feature_matrix_ori2 = (feature_matrix_ori-np.min(feature_matrix_ori))/(np.max(feature_matrix_ori)-np.min(feature_matrix_ori))
        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
       # filename = "/data/jing001/gae-FS/FC_matrix/FC" + str(i) + ".txt"
       # np.savetxt(filename, feature_matrix_ori,fmt='%f',delimiter=',')
        #
        #
        feature_matrix = feature_matrix_ori2#[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        #tem = np.array(np.abs(feature_matrix))
        #feature_matrix[tem<0.2] = 0 
        #feature_matrix = feature_matrix_ori[~np.eye(feature_matrix_ori.shape[0],dtype=bool)].reshape(feature_matrix_ori.shape[0],-1)
       
        #edge_index_coo = coo_matrix(data[i])
        #edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)
        edge_index_coo = np.triu_indices(264, k=1)
        edge_adj = np.zeros((264, 264))
        for ii in range(len(feature_matrix_ori2[1])):
            #index_max = np.argmax(feature_matrix_ori2[ii], n=kk)
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        #print(edge_adj)
        #edge_weight = feature_matrix_ori2[edge_index_coo]
        edge_weight = edge_adj[edge_index_coo]
        #print(edge_weight.shape)
        #target_scaler = preprocessing.StandardScaler().fit(edge_weight.reshape(-1,1))
        #edge_weight = target_scaler.transform(edge_weight.reshape(-1,1))[:,0]    
        #print(edge_weight.shape)
        #edge_weight[edge_weight<0] = 0  ##for FC
        #edge_weight[np.abs(edge_weight)<thres] = 0   ##for FC #0.3
        edge_index_coo = torch.tensor(edge_index_coo)
        #edge_weight = torch.tensor(edge_weight)
       #print(edge_weight)
       # print(edge_index_coo)
        if features != None:
            feature_matrix = features[i][0]
        #print(feature_matrix.shape)
        #print(feature_matrix.shape)
        #print(edge_index_coo.shape)
        #print(edge_weight.shape)
        #print(torch.tensor(index,  dtype = torch.int))
    
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, edge_weight=torch.tensor(edge_weight, dtype = torch.float32), y = torch.tensor(indexx[i]))
        #print(np.reshape(feature_matrix,(1,data.shape[1],data.shape[2])))
        dataset_list.append(graph_data)
    #print(torch.tensor(dataset_list).x)
    #print(dataset_list)
    return dataset_list

def create_fusion_end_dataset(data, data2, score,indexx, features = None):
    dataset_list = []
    n = data.shape[1]
    kk = 11 #noGCN-33#22 17 #preserve top 10% edges 11 best
    for i in range(len(data)):
       # degree_matrix = np.count_nonzero(data[i], axis=1).reshape(n,1)
       # weight_matrix = np.diag(np.sum(data[i], axis=1)).diagonal().reshape(n,1)
        #feature_matrix = np.hstack((degree_matrix, weight_matrix))
        feature_matrix_ori = np.array(data[i])
        feature_matrix_ori_SC = np.array(data2[i])

        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
        feature_matrix_ori_SC2 =  feature_matrix_ori_SC/np.max(feature_matrix_ori_SC)
        feature_matrix = feature_matrix_ori2[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        feature_matrix2 = feature_matrix_ori_SC2[~np.eye(feature_matrix_ori_SC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_SC2.shape[0],-1)
        #print(np.max(node_SC),np.min(node_SC))
        feature_matrix_total = np.concatenate([feature_matrix, feature_matrix2], axis = 0)
        #print(feature_matrix_total.shape)

        edge_index_coo = np.triu_indices(116, k=1)
        edge_index_coo2 = np.triu_indices(116, k=1)
        edge_adj = np.zeros((116, 116))
        for ii in range(len(feature_matrix_ori2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        edge_weight = edge_adj[edge_index_coo]
        edge_weight2 = feature_matrix_ori_SC2[edge_index_coo2]
        edge_weight2[edge_weight2<0.6] = 0  #0.3-noGCNbest
        edge_weight3 = np.ones(116,int)
        edge_weight_total = np.concatenate([edge_weight, edge_weight2, edge_weight3], axis = 0)

        edge_index_coo = torch.tensor(edge_index_coo)
        edge_index_coo2 = torch.tensor(edge_index_coo2)
        edge_index_coo2 = edge_index_coo2+116
        edge_index_coo4=np.reshape(np.arange(0,116), (1,116))
        edge_index_coo5=np.reshape(np.arange(116,232), (1,116))
        edge_index_coo3 = torch.tensor(np.concatenate((edge_index_coo4,edge_index_coo5),axis=0))        
        edge_index_coo_total = torch.cat((edge_index_coo, edge_index_coo2, edge_index_coo3), 1)
         
        graph_data = Data(x = torch.tensor(feature_matrix_total, dtype = torch.float32), edge_index=edge_index_coo_total, edge_weight=torch.tensor(edge_weight_total, dtype = torch.float32),  y = torch.tensor(indexx[i])) 
        dataset_list.append(graph_data)
    return dataset_list

def create_dataset_SC(data, indexx, features = None):
    dataset_list = []
    n = data.shape[0]
   
    for i in range(len(data)):
        #print(i)
       # degree_matrix = np.count_nonzero(data[i], axis=1).reshape(n,1)
       # weight_matrix = np.diag(np.sum(data[i], axis=1)).diagonal().reshape(n,1)
        #feature_matrix = np.hstack((degree_matrix, weight_matrix))
        feature_matrix_ori = np.array(data[i])
        #print(feature_matrix_ori.shape)
        #target_scaler = preprocessing.StandardScaler().fit(feature_matrix_ori.reshape(-1,1))
        #feature_matrix_ori = target_scaler.transform(feature_matrix_ori.reshape(-1,1))[:,0]    
        #feature_matrix_ori = feature_matrix_ori.reshape(116,116)
        #feature_matrix_ori2 = (feature_matrix_ori-np.min(feature_matrix_ori))/(np.max(feature_matrix_ori)-np.min(feature_matrix_ori))
        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
       # filename = "/data/jing001/gae-FS/FC_matrix/FC" + str(i) + ".txt"
       # np.savetxt(filename, feature_matrix_ori,fmt='%f',delimiter=',')
        #
        #
        feature_matrix = feature_matrix_ori2#[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        #tem = np.array(np.abs(feature_matrix))
        #feature_matrix[tem<0.2] = 0 
        #feature_matrix = feature_matrix_ori[~np.eye(feature_matrix_ori.shape[0],dtype=bool)].reshape(feature_matrix_ori.shape[0],-1)
       
        #edge_index_coo = coo_matrix(data[i])
        #edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)
        edge_index_coo = np.triu_indices(264, k=1)
        edge_adj = feature_matrix_ori2
        #for ii in range(len(feature_matrix_ori2[1])):
        #    #index_max = np.argmax(feature_matrix_ori2[ii], n=kk)
        #    index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
        #    edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        #print(edge_adj)
        #edge_weight = feature_matrix_ori2[edge_index_coo]
        edge_weight = edge_adj[edge_index_coo]
        #print(edge_weight.shape)
        #target_scaler = preprocessing.StandardScaler().fit(edge_weight.reshape(-1,1))
        #edge_weight = target_scaler.transform(edge_weight.reshape(-1,1))[:,0]    
        #print(edge_weight.shape)
        #edge_weight[edge_weight<0] = 0  ##for FC
        edge_weight[edge_weight<0.3] = 0   ##for FC #0.3
        edge_index_coo = torch.tensor(edge_index_coo)
        #edge_weight = torch.tensor(edge_weight)
       #print(edge_weight)
       # print(edge_index_coo)
        if features != None:
            feature_matrix = features[i][0]
        #print(feature_matrix.shape)
        #print(feature_matrix.shape)
        #print(edge_index_coo.shape)
        #print(edge_weight.shape)
        #print(torch.tensor(index,  dtype = torch.int))
    
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, edge_weight=torch.tensor(edge_weight, dtype = torch.float32), y = torch.tensor(indexx[i]))
        #print(np.reshape(feature_matrix,(1,data.shape[1],data.shape[2])))
        dataset_list.append(graph_data)
    #print(torch.tensor(dataset_list).x)
    #print(dataset_list)
    return dataset_list
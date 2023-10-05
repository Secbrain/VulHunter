# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.utils.data as data
# from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
import io
from tqdm import tqdm
import numpy as np
import torch
import re
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
from sklearn import metrics
import sklearn
import json

# from decimal import *
# from pyevmasm import disassemble_hex
# from pyevmasm import assemble_hex

# from solc import compile_source, compile_files, link_code
# from solcx import compile_source, compile_files, link_code, get_installed_solc_versions, set_solc_version
# from evm_cfg_builder.cfg import CFG

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import time
import random
import gc

from torch.optim.lr_scheduler import LambdaLR

import math
import torch.nn.functional as F

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmarlk = False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Dataset_Maker(data.Dataset):
    def __init__(self,mode, data_inputs, label_inputs, data_len, kind_size, transform=None, target_transform=None):
        super(Dataset_Maker, self).__init__()
#         self.root = '.' + os.sep+'data/image_cross_PoorToNormal//' + mode
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.data_len = data_len
        self.kind_size = kind_size
        #load subfloders
#         classes = os.listdir(self.root)
        #load image from subfloders
        self.data = np.zeros((len(data_inputs),kind_size,data_len,1));
        self.label = np.zeros(len(data_inputs), dtype=np.int);
        
        if mode == 'Train':
            idx = np.arange(len(data_inputs))
            np.random.shuffle(idx)
            for i in range(len(data_inputs)):
                if len(data_inputs[idx[i]]) <= kind_size*0.4: #去掉少于4项样例
                    continue
                for k in range(min(kind_size, len(data_inputs[idx[i]]))):
                    for j in range(min(data_len, len(data_inputs[idx[i]][k]))):
                        self.data[i][k][j][0] = data_inputs[idx[i]][k][j]
                self.label[i] = label_inputs[idx[i]]
        else:
            for i in range(len(data_inputs)):
                if len(data_inputs[i]) <= kind_size*0.4:
                    continue
                for k in range(min(kind_size, len(data_inputs[i]))):
                    for j in range(min(data_len, len(data_inputs[i][k]))):
                        self.data[i][k][j][0] = data_inputs[i][k][j]
                self.label[i] = label_inputs[i]
        
        print(mode+ 'set len :'+str(len(self.data)))

                
    def __getitem__(self, idx):
        # print(idx,len(self.data))
        x = self.data[idx]
#         x = x.reshape((1,self.data_len,self.kind_size))
        if True:
#             x = self.transform(x)
            x = torch.from_numpy(x).float()
            
        return x, self.label[idx]

    def __len__(self):
        return len(self.data)

class Dataset_Maker_Mulins(data.Dataset):
    def __init__(self,mode, data_inputs, label_inputs, data_len, kind_size, transform=None, target_transform=None):
        super(Dataset_Maker_Mulins, self).__init__()
#         self.root = '.' + os.sep+'data/image_cross_PoorToNormal//' + mode
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.data_len = data_len
        self.kind_size = kind_size
        #load subfloders
#         classes = os.listdir(self.root)
        #load image from subfloders
        self.data = []
        self.label = []

        # self.databag = []
        # self.labelbag = []

        self.bag = []
        
        if mode == 'Train':
            idx = np.arange(len(data_inputs))
            np.random.shuffle(idx)
            for i in range(len(data_inputs)):
                if len(data_inputs[idx[i]]) <= kind_size*0.4: #去掉少于4项样例
                    continue
                list_val = []
                for k in range(min(kind_size, len(data_inputs[idx[i]]))):
                    data_val = np.zeros((data_len,1))
                    for j in range(min(data_len, len(data_inputs[idx[i]][k]))):
                        data_val[j][0] = data_inputs[idx[i]][k][j]
                    self.data.append(data_val)
                    self.label.append(label_inputs[idx[i]])
                    list_val.append(data_val)
                    self.bag.append(i)
                # self.databag.append(list_val)
                # self.labelbag.append(label_inputs[idx[i]])
        else:
            for i in range(len(data_inputs)):
                # if len(data_inputs[i]) <= kind_size*0.4:
                #     continue
                list_val = []
                for k in range(min(kind_size, len(data_inputs[i]))):
                    data_val = np.zeros((data_len,1))
                    for j in range(min(data_len, len(data_inputs[i][k]))):
                        data_val[j][0] = data_inputs[i][k][j]
                    self.data.append(data_val)
                    self.label.append(label_inputs[i])
                    list_val.append(data_val)
                    self.bag.append(i)
                # self.databag.append(list_val)
                # self.labelbag[i] = label_inputs[i]
            
        self.data = np.array(self.data, dtype=np.float)
        self.label = np.array(self.label)
        self.bag = np.array(self.bag, dtype=np.int)
        # self.databag = np.array(self.databag, dtype=np.float)
        # self.labelbag = np.array(self.labelbag, dtype=np.int)

        # print(mode+ 'set len :'+str(len(self.databag)))
        print(mode+ 'set subitem len :'+str(len(self.data)))

    def __getitem__(self, idx):
        # print(idx,len(self.data))
        x = self.data[idx]
#         x = x.reshape((1,self.data_len,self.kind_size))
        if True:
#             x = self.transform(x)
            x = torch.from_numpy(x).float()
            
        return x, self.label[idx], self.bag[idx]

    def __len__(self):
        return len(self.data)

class Dataset_Maker_Mulins_pre(data.Dataset):
    def __init__(self,mode, data_inputs, data_len, kind_size, transform=None, target_transform=None):
        super(Dataset_Maker_Mulins_pre, self).__init__()
#         self.root = '.' + os.sep+'data/image_cross_PoorToNormal//' + mode
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.data_len = data_len
        self.kind_size = kind_size
        #load subfloders
#         classes = os.listdir(self.root)
        #load image from subfloders
        self.data = []

        # self.databag = []
        # self.labelbag = []

        self.bag = []
        
        if mode == 'Train':
            idx = np.arange(len(data_inputs))
            np.random.shuffle(idx)
            for i in range(len(data_inputs)):
                # if len(data_inputs[idx[i]]) <= kind_size*0.4: #去掉少于4项样例
                #     continue
                list_val = []
                for k in range(min(kind_size, len(data_inputs[idx[i]]))):
                    data_val = np.zeros((data_len,1))
                    for j in range(min(data_len, len(data_inputs[idx[i]][k]))):
                        data_val[j][0] = data_inputs[idx[i]][k][j]
                    self.data.append(data_val)
                    list_val.append(data_val)
                    self.bag.append(i)
                # self.databag.append(list_val)
                # self.labelbag.append(label_inputs[idx[i]])
        else:
            for i in range(len(data_inputs)):
                # if len(data_inputs[i]) <= kind_size*0.4:
                #     continue
                list_val = []
                for k in range(min(kind_size, len(data_inputs[i]))):
                    data_val = np.zeros((data_len,1))
                    for j in range(min(data_len, len(data_inputs[i][k]))):
                        data_val[j][0] = data_inputs[i][k][j]
                    self.data.append(data_val)
                    list_val.append(data_val)
                    self.bag.append(i)
                # self.databag.append(list_val)
                # self.labelbag[i] = label_inputs[i]
            
        self.data = np.array(self.data, dtype=np.float)
        self.bag = np.array(self.bag, dtype=np.int)
        # self.databag = np.array(self.databag, dtype=np.float)
        # self.labelbag = np.array(self.labelbag, dtype=np.int)

        # print(mode+ 'set len :'+str(len(self.databag)))
        print(mode+ 'set subitem len :'+str(len(self.data)))

    def __getitem__(self, idx):
        # print(idx,len(self.data))
        x = self.data[idx]
#         x = x.reshape((1,self.data_len,self.kind_size))
        if True:
#             x = self.transform(x)
            x = torch.from_numpy(x).float()
            
        return x, self.bag[idx]

    def __len__(self):
        return len(self.data)

class Dataset_Maker_Mulins_Upd(data.Dataset):
    def __init__(self, data_inputs, label_inputs, bag_inputs, transform=None, target_transform=None):
        super(Dataset_Maker_Mulins_Upd, self).__init__()
#         self.root = '.' + os.sep+'data/image_cross_PoorToNormal//' + mode
        self.transform = transform
        self.target_transform = target_transform
        #load subfloders
#         classes = os.listdir(self.root)
        #load image from subfloders
        self.data = []
        self.label = []
        self.bag = []
        
        print("update datasets maker")
        # idx = np.arange(len(data_inputs))
        # np.random.shuffle(idx)
        
        self.data = data_inputs
        self.label = label_inputs
        self.bag = bag_inputs

        # self.data = np.array(self.data, dtype=np.float)
        # self.label = np.array(self.label, dtype=np.int)
        # self.bag = np.array(self.bag, dtype=np.int)

    def __getitem__(self, idx):
        # print(idx,len(self.data))
        x = self.data[idx]
#         x = x.reshape((1,self.data_len,self.kind_size))
        if True:
#             x = self.transform(x)
            x = torch.from_numpy(x).float()
            
        return x, self.label[idx], self.bag[idx]

    def __len__(self):
        return len(self.data)

class ProtoNet_Attention(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=512, z_dim=64):
        super(ProtoNet_Attention, self).__init__()
        input_size = 1
        hidden_size = hid_dim
        num_layer = 1
        
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        
        )
        
        self.linear1 =  nn.Linear(7*7*256,256)
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size,hidden_size,num_layer,batch_first=True,bidirectional=True) #,bidirectional=True
        )
        self.GRU = nn.Sequential(
            nn.GRU(input_size,hidden_size,num_layer,batch_first=True,bidirectional=True) #,bidirectional=True,LSTM
        )

        self.Linear = nn.Sequential(
            nn.Linear(512*2,512), #*2 *hid_dim
            nn.Linear(512,64),
            nn.Linear(64,16),
            # nn.Linear(16,4),
            nn.Linear(16,2),
#             nn.Linear(4,2), #二分类
            nn.Softmax()
        )
        self.fc = nn.Linear(512*2, 2)
        self.fc1 = nn.Linear(512*512*2, 2)
        self.dropout = nn.Dropout(0.5)

    # x: [batch, seq_len, hidden_dim*2]
    # query : [batch, seq_len, hidden_dim * 2]
    # 软注意力机制 (key=value=x)
#     def attention_net(self, x, query, mask=None): 
        
#         d_k = query.size(-1)     # d_k为query的维度
       
#         # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
# #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
#         # 打分机制 scores: [batch, seq_len, seq_len]
#         scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
# #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
#         # 对最后一个维度 归一化得分
#         alpha_n = F.softmax(scores, dim=-1) 
# #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
#         # 对权重化的x求和
#         # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
#         context = torch.matmul(alpha_n, x).sum(1)
        
#         return context, alpha_n

    def attention_net(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
#         x = self.conv1(x)
        
#         x = self.conv2(x)
#         x = self.linear1(x.view(x.size(0), -1))
        x = x.view((-1,512,1)) #batch,长度,纬度
        # x,_ = self.GRU(x)

        #注意力
        x, (hidden, cell) = self.LSTM(x)
        # attn_output = self.attention(x, hidden)
        # query = self.dropout(x)
        # attn_output, alpha_n = self.attention_net(x, query)
        # print(attn_output.size()) # size hidden*2 *1 union

        s,b,h = x.size()
        x = x.reshape(s,b*h)
        # print("x", x.shape)
        x = self.fc1(x) #Linear
        # print("returnx", x.shape)
        
#         print("{} {} {}".format(s,b,h))
        
        # x = self.fc1(x)
        # print(x)
        return x

max_len = 512
batchsize = 512 #512
kind_size = 10
epoch_num = 50

# data_save_dir = './data_shape10'
# model_save_dir = './LSTM_models_mulins'

# data_save_dir = '/home/zhao/contracts/small_bytecode/small_datasets_data_train_test'
data_save_dir = '/home/zhao/contracts/small_bytecode/small_dataset_test_new'
# data_save_dir = '/home/zhao/contracts/small_bytecode/small_datasets_data_alltest'
# model_save_dir = '/home/zhao/contracts/LSTM_models_mulins'
model_save_dir = '/home/smartcontract/SmartContract/vulhunter/vulhunter_code_cpu/models'

result_save_dir = '/home/zhao/contracts/small_bytecode/result_small_bytecode'

data_use_name = ''
vul_use_name = ''

def data_making_all():
    print("{} processing!".format(vul_use_name))

    files_bytecodes = None
    with open('/home/zhao/contracts/small_bytecode/contract_bytecode3_list10_continue2_small_bytedataset.json') as fp:
        files_bytecodes = json.load(fp)
        
    reentrancy_files = None
    with open('/home/zhao/contracts/small_bytecode/labels/' + data_use_name + '.json') as fp:
        reentrancy_files = json.load(fp)

    benign_bytecodes = []
    reentrancy_bytecodes = []
    for k,v in files_bytecodes.items():
        k = os.path.splitext(k)[0]
        if (k in reentrancy_files) and (reentrancy_files[k] == 1):
            reentrancy_bytecodes.append((k,v))
        else:
            benign_bytecodes.append((k,v))

    print("benign: {}, malicious {}".format(len(benign_bytecodes), len(reentrancy_bytecodes)))

    random.shuffle(benign_bytecodes)
    random.shuffle(reentrancy_bytecodes)
    train_bytecodes = []
    test_bytecodes = []
    train_labels = []
    test_labels = []
    train_bytecodes = benign_bytecodes[:int(len(benign_bytecodes)*0.8)] + reentrancy_bytecodes[:int(len(reentrancy_bytecodes)*0.8)]
    train_labels = [0] * int(len(benign_bytecodes)*0.8) + [1] * int(len(train_bytecodes) - int(len(benign_bytecodes)*0.8))
    test_bytecodes = benign_bytecodes[int(len(benign_bytecodes)*0.8):int(len(benign_bytecodes))] + reentrancy_bytecodes[int(len(reentrancy_bytecodes)*0.8):]
    test_labels = [0] * int(int(len(benign_bytecodes)) - int(len(benign_bytecodes)*0.8)) + [1] * int(len(test_bytecodes) - int(int(len(benign_bytecodes)) - int(len(benign_bytecodes)*0.8)))
    
    train_names = []
    test_names = []

    train_names = [interval[0] for interval in train_bytecodes]
    test_names = [interval[0] for interval in test_bytecodes]
    train_bytecodes = [interval[1] for interval in train_bytecodes]
    test_bytecodes = [interval[1] for interval in test_bytecodes]

    print('data length:')
    print(len(train_bytecodes))
    print(len(train_labels))
    print(len(test_bytecodes))
    print(len(test_labels))

    return train_bytecodes, train_labels, test_bytecodes, test_labels, train_names, test_names

def making_all_dataset():
    print("{} processing!".format(vul_use_name))

    files_bytecodes = None
    with open('/home/zhao/contracts/small_bytecode/contract_bytecode3_list10_continue2_small_bytedataset.json') as fp:
        files_bytecodes = json.load(fp)
        
    reentrancy_files = None
    with open('/home/zhao/contracts/small_bytecode/labels/' + vul_use_name + '.json') as fp:
        reentrancy_files = json.load(fp)

    benign_bytecodes = []
    reentrancy_bytecodes = []
    for k,v in files_bytecodes.items():
        k = os.path.splitext(k)[0]
        if (k in reentrancy_files) and (reentrancy_files[k] == 1):
            reentrancy_bytecodes.append((k,v))
        else:
            benign_bytecodes.append((k,v))

    print("benign: {}, malicious {}".format(len(benign_bytecodes), len(reentrancy_bytecodes)))

    random.shuffle(benign_bytecodes)
    random.shuffle(reentrancy_bytecodes)

    test_bytecodes = []
    test_labels = []

    test_bytecodes = benign_bytecodes + reentrancy_bytecodes
    test_labels = [0] * int(len(benign_bytecodes)) + [1] * int(len(reentrancy_bytecodes))
    
    test_names = []

    test_names = [interval[0] for interval in test_bytecodes]
    test_bytecodes = [interval[1] for interval in test_bytecodes]


    print('data length:')

    print(len(test_bytecodes))
    print(len(test_labels))

    return test_bytecodes, test_labels, test_names

def data_making():
    print("{} processing!".format(vul_use_name))
    files_bytecodes = None
    with open('/home/zhao/contracts/small_bytecode/contract_bytecode3_list10_continue2_small_bytedataset.json') as fp:
        files_bytecodes = json.load(fp)
        
    reentrancy_files = None
    with open('/home/zhao/contracts/small_bytecode/labels/' + data_use_name + '.json') as fp:
        reentrancy_files = json.load(fp)

    benign_bytecodes = []
    reentrancy_bytecodes = []
    for k,v in files_bytecodes.items():
        k = os.path.splitext(k)[0]
        if (k in reentrancy_files) and (reentrancy_files[k] == 1):
            reentrancy_bytecodes.append((k,v))
        else:
            benign_bytecodes.append((k,v))

    print("benign: {}, malicious {}".format(len(benign_bytecodes), len(reentrancy_bytecodes)))

    random.shuffle(benign_bytecodes)
    random.shuffle(reentrancy_bytecodes)
    train_bytecodes = []
    test_bytecodes = []
    train_labels = []
    test_labels = []
    ratio =  (len(reentrancy_bytecodes) * 1.0) / (len(benign_bytecodes) * 1.0) * 2.0
    train_bytecodes = benign_bytecodes[:int(min(len(benign_bytecodes), len(benign_bytecodes)*ratio)*0.8)] + reentrancy_bytecodes[:int(len(reentrancy_bytecodes)*0.8)]
    train_labels = [0] * int(len(train_bytecodes) - int(len(reentrancy_bytecodes)*0.8)) + [1] * int(len(reentrancy_bytecodes)*0.8)
    test_bytecodes = benign_bytecodes[int(min(len(benign_bytecodes), len(benign_bytecodes)*ratio)*0.8):int(min(len(benign_bytecodes), len(benign_bytecodes)*ratio))] + reentrancy_bytecodes[int(len(reentrancy_bytecodes)*0.8):]
    test_labels = [0] * int(len(test_bytecodes) - len(reentrancy_bytecodes) + int(len(reentrancy_bytecodes)*0.8)) + [1] * int(len(reentrancy_bytecodes)-int(len(reentrancy_bytecodes)*0.8))
    
    train_names = []
    test_names = []

    train_names = [interval[0] for interval in train_bytecodes]
    test_names = [interval[0] for interval in test_bytecodes]
    train_bytecodes = [interval[1] for interval in train_bytecodes]
    test_bytecodes = [interval[1] for interval in test_bytecodes]

    print('data length:')
    print(len(train_bytecodes))
    print(len(train_labels))
    print(len(test_bytecodes))
    print(len(test_labels))

    return train_bytecodes, train_labels, test_bytecodes, test_labels, train_names, test_names


def train_test_LSTM_with_multi_instance(train_bytecodes, train_labels, test_bytecodes, test_labels):
    
    trainset = Dataset_Maker_Mulins('Train', train_bytecodes, train_labels, max_len, kind_size)
    testset = Dataset_Maker_Mulins('Test', test_bytecodes, test_labels, max_len, kind_size)

    DataLoader_train = DataLoader(trainset,batch_size=batchsize,shuffle=False)
    DataLoader_test = DataLoader(testset,batch_size=batchsize,shuffle=False)

    model = ProtoNet_Attention()
    if torch.cuda.is_available():
        print("using cuda!")
        model = model.cuda() #ProtoNet
        if torch.cuda.device_count() >= 2:
            print("using devices num {}".format(torch.cuda.device_count()))
            model=torch.nn.DataParallel(model)

    #交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    #momentum 冲量
    # optimizer = optim.SGD(model.parameters(),lr=0.01)
    # optimizer = optim.RAdam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0012, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.1*epoch)

    def train(epoch, dataloader):
        running_loss = 0.0
        # print("lr: {}".format(scheduler.get_lr()))
    #     print(type(DataLoader_train))
        for batch_idx,data in enumerate(dataloader):
            inputs,target,bags = data
            if torch.cuda.is_available():
                inputs = inputs.cuda().float()
                target = target.cuda().long()
            else:
                inputs = inputs.float()
                target = target.long()
            optimizer.zero_grad()  #梯度值清零
            outputs = model(inputs)   #前向传播
            # print(outputs)
            # print(type(outputs))
            bias = torch.sub(outputs.data[:,0], outputs.data[:,1], alpha=1)
            bag_index = {}
            bags = bags.numpy()
            for i in range(len(bags)):
                if bags[i] not in bag_index:
                    bag_index[bags[i]] = []
                bag_index[bags[i]].append(i)
            
            # print(bag_index)
            # predicted = predicted.cpu().numpy
            # outputs = outputs.cpu()
            # target = target.numpy()
            true_label_index = []
            pre_index = []
            for bag_val, index_val in bag_index.items():
                true_label_index.append(index_val[0])
                # print(bias[index_val].size())
                pre_index.append(index_val[torch.argmin(bias[index_val])])
                # pre_label.append(outputs[index_val][pre_index])
            # print(outputs.cpu().detach().numpy())
            # pre_label = outputs[pre_index]
            # target = target[true_label_index]
            # print(pre_label.size())
            # print(target.size())
            loss = criterion(outputs[pre_index],target[true_label_index]) + 0.6 * criterion(outputs,target)
            loss.backward()   #反向传播
            optimizer.step()  #更新权重

            # del bag_index
            # gc.collect()

            running_loss+=loss.item()
            if batch_idx %10==9:
                loss = running_loss/10
                # print('[%d,%5d] loss:%.3f' %(epoch+1,batch_idx+1,loss)) #输出loss---
                if running_loss < 1e-4:
                    break
                running_loss=0.0
            
            # loss = validate_loss(validdataloader)
            # loss.backward()   #反向传播
            # optimizer.step()  #更新权重
            # print('[%d,%5d] loss:%.3f' %(epoch+1,i+1,loss))
            # # scheduler.step()

    def updata_data(dataloader):
        print("update data")
        # print("lr: {}".format(scheduler.get_lr()))
    #     print(type(DataLoader_train))
        datasets_data = []
        class_bias = []
        datasets_labels = []
        datasets_bags = []

        with torch.no_grad():   #使得以下代码执行过程中不用求梯度
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                datasets_data.extend(inputs.numpy())
                datasets_labels.extend(labels.numpy())
                datasets_bags.extend(bags.numpy())

                if torch.cuda.is_available():
                    inputs = inputs.cuda().float()
                    labels = labels.cuda().long()
                else:
                    inputs = inputs.float()
                    labels = labels.long()
                outputs = model(inputs)
                bias = torch.sub(outputs.data[:,0], outputs.data[:,1], alpha=1)
                
                if torch.cuda.is_available():
                    class_bias.extend(bias.cpu().numpy())
                else:
                    class_bias.extend(bias.numpy())

        print("end of predict")
        datasets_data = np.array(datasets_data)
        datasets_labels = np.array(datasets_labels)
        datasets_bags = np.array(datasets_bags)
        df = pd.DataFrame({'bags': datasets_bags, 'labels': datasets_labels, 'bias': class_bias})
        group = df.groupby('bags')
        indexlist = []
        for bag,x in group:
            if(x['labels'].values[0]==0):
                # indexlist.extend(x['bias'].sort_values(ascending=False, inplace=False).index.tolist()[:int((len(x)+1)*0.8)])
                # indexlist.append(x['bias'].idxmax())
                indexlist.extend(x.index.tolist())
            else:
                indexlist.extend(x['bias'].sort_values(ascending=False, inplace=False).index.tolist()[int((len(x)+1)*0.8):])
                # indexlist.append(x['bias'].idxmin())
        del df, group
        gc.collect()
        print("data length: {}".format(len(indexlist)))
        return datasets_data[indexlist], datasets_labels[indexlist], datasets_bags[indexlist]
    
    def test(dataloader):
        # correct = 0
        # total = 0
        print("test datasets")
        true_labels = []
        predict_labels = []
        class_bags = []
        with torch.no_grad():   #使得以下代码执行过程中不用求梯度
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda().float()
                    labels = labels.cuda().long()
                else:
                    inputs = inputs.float()
                    labels = labels.long()
                
                outputs = model(inputs)
                # 在每一行中求最大值的下标，返回两个参数，第一个为最大值，第二个为坐标
                #dim=1 数值方向的维度为0，水平方向的维度为1
                # print(outputs.data)
                _,predicted = torch.max(outputs.data,dim=1)
                # total+=labels.size(0)
                #当预测值与标签相同时取出并求和

                if torch.cuda.is_available():
                    predict_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                else:
                    predict_labels.extend(predicted.numpy())
                    true_labels.extend(labels.numpy())
                
                class_bags.extend(bags.numpy())
                # correct+=(predicted==labels).sum().item()
        
        predict_labels = np.array(predict_labels)
        true_labels = np.array(true_labels)
        class_bags = np.array(class_bags)
        df = pd.DataFrame({'bags': class_bags, 'labels': true_labels, 'prelabels': predict_labels})
        group = df.groupby('bags')
        true_labels = []
        predict_labels = []

        for bag,x in group:
            true_labels.append(x['labels'].values[0])
            if (x['prelabels']==1).any():
                predict_labels.append(1)
            else:
                predict_labels.append(0)

    #     print('Accuracy on test set:%d %%' % (100*correct/total))
        del df, group
        gc.collect()
        # print(predict_labels)
        return true_labels, predict_labels

    best_epoch = 0
    best_acc = 0
    best_recall = 0
    best_precision = 0
    best_f1 = 0
    DataLoader_train_upd = None
    for epoch in range(epoch_num):
        model.train()
        if epoch == 0:
            train(epoch,DataLoader_train)
        else:
            train(epoch,DataLoader_train_upd)
        model.eval()
        new_datas, new_labels, new_bags = updata_data(DataLoader_train)
        print("ceshi!")
        trainset_upd = Dataset_Maker_Mulins_Upd(new_datas, new_labels, new_bags)
        DataLoader_train_upd = DataLoader(trainset_upd,batch_size=batchsize,shuffle=False)
        
        del new_datas, new_labels, trainset_upd
        gc.collect()

        true_labels, predict_labels = test(DataLoader_test)

        acc = metrics.accuracy_score(true_labels, predict_labels)
        recall = metrics.recall_score(true_labels, predict_labels, average="binary")
        precision = metrics.precision_score(true_labels, predict_labels, average="binary")
        F1 = metrics.f1_score(true_labels, predict_labels, average="binary")

        print("{} Accuracy on test set: {}, recall: {}, precision: {}, f1: {}".format(epoch, acc, recall, precision, F1))
        if best_acc <= acc:
            best_acc = acc
            best_recall = recall
            best_precision = precision
            best_f1 = F1
            best_epoch = epoch
            # 保存
            if torch.cuda.device_count() >= 2:
                torch.save(model.module.state_dict(), os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth"))
            else:
                torch.save(model.state_dict(), os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth"))
            print("model save {}!".format(model_save_dir))
        print("Best accuracy on test set, epoch: {}, accuracy: {}, recall: {}, precision: {}, f1: {}".format(best_epoch, best_acc, best_recall, best_precision, best_f1))

        del true_labels, predict_labels
        gc.collect()

def test_LSTM_with_multi_instance(test_bytecodes, test_labels):
        
    testset = Dataset_Maker_Mulins('Test', test_bytecodes, test_labels, max_len, kind_size)

    DataLoader_test = DataLoader(testset,batch_size=batchsize,shuffle=False)

    model = ProtoNet_Attention()
    # torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
    
    model.load_state_dict(torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth")))
    # model = torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
    if torch.cuda.is_available():
        print("using cuda!")
        model = model.cuda() #ProtoNet
        if torch.cuda.device_count() >= 2:
            print("using devices num {}".format(torch.cuda.device_count()))
            model=torch.nn.DataParallel(model)

    def test(dataloader):
        # correct = 0
        # total = 0
        print("test datasets")
        true_labels = []
        predict_labels = []
        class_bags = []
        with torch.no_grad():   #使得以下代码执行过程中不用求梯度
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                inputs = inputs.cuda().float()
                labels = labels.cuda().long()
                
                outputs = model(inputs)
                # 在每一行中求最大值的下标，返回两个参数，第一个为最大值，第二个为坐标
                #dim=1 数值方向的维度为0，水平方向的维度为1
                # print(outputs.data)
                _,predicted = torch.max(outputs.data,dim=1)
                # total+=labels.size(0)
                #当预测值与标签相同时取出并求和
                predict_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                class_bags.extend(bags.numpy())
                # correct+=(predicted==labels).sum().item()
        
        predict_labels = np.array(predict_labels)
        true_labels = np.array(true_labels)
        class_bags = np.array(class_bags)
        df = pd.DataFrame({'bags': class_bags, 'labels': true_labels, 'prelabels': predict_labels})
        group = df.groupby('bags')
        true_labels = []
        predict_labels = []

        for bag,x in group:
            true_labels.append(x['labels'].values[0])
            if (x['prelabels']==1).any():
                predict_labels.append(1)
            else:
                predict_labels.append(0)

    #     print('Accuracy on test set:%d %%' % (100*correct/total))
        del df, group
        gc.collect()
        # print(predict_labels)
        return true_labels, predict_labels
    
    model.eval()
    true_labels, predict_labels = test(DataLoader_test)
    # print(predict_labels[:50])

    acc = metrics.accuracy_score(true_labels, predict_labels)
    recall = metrics.recall_score(true_labels, predict_labels, average="binary")
    precision = metrics.precision_score(true_labels, predict_labels, average="binary")
    F1 = metrics.f1_score(true_labels, predict_labels, average="binary")

    print("Accuracy on test set: {}, recall: {}, precision: {}, f1: {}".format(acc, recall, precision, F1))

    del true_labels, predict_labels
    gc.collect()

def test_LSTM_with_multi_instance_prelabel(test_bytecodes, test_names):
        
    testset = Dataset_Maker_Mulins('Test', test_bytecodes, test_names, max_len, kind_size)
    DataLoader_test = DataLoader(testset,batch_size=batchsize,shuffle=False)

    model = ProtoNet_Attention()
    # torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
    
    model.load_state_dict(torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth")))
    # model = torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
    if torch.cuda.is_available():
        print("using cuda!")
        model = model.cuda() #ProtoNet
        if torch.cuda.device_count() >= 2:
            print("using devices num {}".format(torch.cuda.device_count()))
            model=torch.nn.DataParallel(model)

    def test(dataloader):
        # correct = 0
        # total = 0
        print("test datasets")
        true_labels = []
        predict_labels = []
        class_bags = []
        with torch.no_grad():   #使得以下代码执行过程中不用求梯度
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                inputs = inputs.cuda().float()
                # labels = labels.cuda().long()
                labels = list(labels)
                
                outputs = model(inputs)
                # 在每一行中求最大值的下标，返回两个参数，第一个为最大值，第二个为坐标
                #dim=1 数值方向的维度为0，水平方向的维度为1
                # print(outputs.data)
                _,predicted = torch.max(outputs.data,dim=1)
                # total+=labels.size(0)
                #当预测值与标签相同时取出并求和
                predict_labels.extend(predicted.cpu().numpy())
                # true_labels.extend(labels.cpu().numpy())
                true_labels.extend(labels)
                class_bags.extend(bags.numpy())
                # correct+=(predicted==labels).sum().item()
        
        predict_labels = np.array(predict_labels)
        true_labels = np.array(true_labels)
        class_bags = np.array(class_bags)
        df = pd.DataFrame({'bags': class_bags, 'labels': true_labels, 'prelabels': predict_labels})
        group = df.groupby('bags')
        true_labels = []
        predict_labels = []
        instances_predict_labels = []

        for bag,x in group:
            # true_labels.append(x['labels'].values[0])
            true_labels.append(x['labels'].values.tolist()[0])
            instances_predict_labels.append(x['prelabels'].values.tolist())
            if (x['prelabels']==1).any():
                predict_labels.append(1)
            else:
                predict_labels.append(0)

    #     print('Accuracy on test set:%d %%' % (100*correct/total))
        del df, group
        gc.collect()
        # print(predict_labels)
        # return true_labels, predict_labels
        return true_labels, predict_labels, instances_predict_labels
    
    model.eval()
    file_label = {}
    # true_labels, predict_labels = test(DataLoader_test)
    true_labels, predict_labels, instances_predict_labels = test(DataLoader_test)
    # print(predict_labels[:50])
    for i in range(len(true_labels)):
        file_label[true_labels[i]] = (predict_labels[i], instances_predict_labels[i])

    # acc = metrics.accuracy_score(true_labels, predict_labels)
    # recall = metrics.recall_score(true_labels, predict_labels, average="binary")
    # precision = metrics.precision_score(true_labels, predict_labels, average="binary")
    # F1 = metrics.f1_score(true_labels, predict_labels, average="binary")

    # print("Accuracy on test set: {}, recall: {}, precision: {}, f1: {}".format(acc, recall, precision, F1))
    data_json = json.dumps(file_label, cls=NpEncoder)
    fileObject = open(os.path.join(result_save_dir, vul_use_name + '_result.json'), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()
    print("{} save successfully!".format(vul_use_name + '_result.json'))

    # del true_labels, predict_labels
    del true_labels, predict_labels, instances_predict_labels, model, file_label
    gc.collect()

def Update_model_save():
    # first delete repeate data, use this one

    model = torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
    # if torch.cuda.is_available():
    #     print("using cuda!")
    #     model = model.cuda() #ProtoNet
    #     if torch.cuda.device_count() >= 2:
    #         print("using devices num {}".format(torch.cuda.device_count()))
    #         model=torch.nn.DataParallel(model) #, device_ids = [0, 1, 2]

    # torch.save(model.module.state_dict(), os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth")) #multi graphics card
    torch.save(model.state_dict(), os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth"))
    print("{} have been saved! path: {}".format(vul_use_name, os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth")))

def train_vul_models(contract_labels_path, bytecodes_path, vul_use_names, model_dir, epoch, max_len_set, batchsize_set, kind_size_set):
    global vul_use_name, model_save_dir, epoch_num, max_len, batchsize, kind_size

    model_save_dir = model_dir
    epoch_num = epoch

    max_len = max_len_set
    batchsize = batchsize_set
    kind_size = kind_size_set

    file_data = None
    with open(contract_labels_path) as fp:
        file_data = json.load(fp)

    files_bytecodes = None
    with open(bytecodes_path) as fp:
        files_bytecodes = json.load(fp)

    for i in range(len(vul_use_names)):
        vul_use_name = vul_use_names[i]
        print("{} working!".format(vul_use_name))

        train_bytecodes = []
        test_bytecodes = []
        train_labels_new = []
        test_labels_new = []
        train_labels = file_data[vul_use_name]['train_labels']
        test_labels = file_data[vul_use_name]['test_labels']
        train_names = file_data[vul_use_name]['train_names']
        test_names = file_data[vul_use_name]['test_names']

        for i in range(len(train_names)):
            if train_names[i] not in files_bytecodes:
                print(train_names[i], "is not outputed normally!")
                continue
            train_bytecodes.append(files_bytecodes[train_names[i]])
            train_labels_new.append(train_labels[i])

        for i in range(len(test_names)):
            if test_names[i] not in files_bytecodes:
                print(test_names[i], "is not outputed normally!")
                continue
            test_bytecodes.append(files_bytecodes[test_names[i]])
            test_labels_new.append(test_labels[i])

        print("train_bytecode", len(train_bytecodes))
        print("test_bytecode", len(test_bytecodes))
        print("train_label", len(train_labels_new))
        print("test_label", len(test_labels_new))

        print('data length:')
        print(len(train_bytecodes))
        print(len(train_labels))
        print(len(test_bytecodes))
        print(len(test_labels))

        print('data load successfully!')

        train_test_LSTM_with_multi_instance(train_bytecodes, train_labels_new, test_bytecodes, test_labels_new)

def test_LSTM_prelabel(test_bytecodes, vul_use_names, model_save_dir, kind_size_set, max_len_set, batchsize_set):
    global kind_size

    max_len = max_len_set
    batchsize = batchsize_set
    kind_size = kind_size_set

    testset = Dataset_Maker_Mulins_pre('Test', test_bytecodes, max_len, kind_size)
    DataLoader_test = DataLoader(testset,batch_size=batchsize,shuffle=False)

    predict_results = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for vul_use_name in vul_use_names:
        print(vul_use_name + ' precessing!')
        
        model = ProtoNet_Attention()
        # torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
        
        # model.load_state_dict(torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth")))
        model.load_state_dict(torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pth"), map_location=device))
        # model = torch.load(os.path.join(model_save_dir, "torch_" + vul_use_name + ".pkl"))
        if torch.cuda.is_available():
            print("using cuda!")
            model = model.cuda() #ProtoNet
            if torch.cuda.device_count() >= 2:
                print("using devices num {}".format(torch.cuda.device_count()))
                model=torch.nn.DataParallel(model)

        def test(dataloader):
            # print("test datasets")
            predict_labels = []
            class_bags = []
            with torch.no_grad():   #使得以下代码执行过程中不用求梯度
                for batch_idx,data in enumerate(dataloader):
                    inputs,bags = data
                    if torch.cuda.is_available():
                        inputs = inputs.cuda().float()
                    else:
                        inputs = inputs.float()
                    # labels = labels.cuda().long()                
                    outputs = model(inputs)
                    # print(outputs.data)
                    # 在每一行中求最大值的下标，返回两个参数，第一个为最大值，第二个为坐标
                    #dim=1 数值方向的维度为0，水平方向的维度为1
                    # print(outputs.data)
                    _,predicted = torch.max(outputs.data,dim=1)
                    # total+=labels.size(0)
                    #当预测值与标签相同时取出并求和

                    if torch.cuda.is_available():
                        predict_labels.extend(predicted.cpu().numpy())
                        # true_labels.extend(labels.cpu().numpy())
                        class_bags.extend(bags.cpu().numpy())
                        # correct+=(predicted==labels).sum().item()
                    else:
                        predict_labels.extend(predicted.numpy())
                        # true_labels.extend(labels.cpu().numpy())
                        class_bags.extend(bags.numpy())
                        # correct+=(predicted==labels).sum().item()
            
            predict_labels = np.array(predict_labels)
            class_bags = np.array(class_bags)
            df = pd.DataFrame({'bags': class_bags, 'prelabels': predict_labels})
            group = df.groupby('bags')
            predict_labels = []
            instances_predict_labels = []

            for bag,x in group:
                instances_predict_labels.append(x['prelabels'].values.tolist())
                if (x['prelabels']==1).any():
                    predict_labels.append(1)
                else:
                    predict_labels.append(0)

        #     print('Accuracy on test set:%d %%' % (100*correct/total))
            del df, group
            gc.collect()
            # print(predict_labels)
            # return true_labels, predict_labels
            return predict_labels, instances_predict_labels
        
        model.eval()
        # true_labels, predict_labels = test(DataLoader_test)
        predict_labels, instances_predict_labels = test(DataLoader_test)

        predict_results[vul_use_name] = {'predict_labels': predict_labels, 'instances_predict_labels':instances_predict_labels}

        del predict_labels, model, instances_predict_labels
        gc.collect()

    return predict_results
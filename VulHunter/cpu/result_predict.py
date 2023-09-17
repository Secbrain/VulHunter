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
                if len(data_inputs[idx[i]]) <= kind_size*0.4: 
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
#             nn.Linear(4,2),
            nn.Softmax()
        )
        self.fc = nn.Linear(512*2, 2)
        self.fc1 = nn.Linear(512*512*2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
#         x = self.conv1(x)
        
#         x = self.conv2(x)
#         x = self.linear1(x.view(x.size(0), -1))
        x = x.view((-1,512,1)) #batch,长度,纬度
        # x,_ = self.GRU(x)

        x, (hidden, cell) = self.LSTM(x)
        attn_output = self.attention(x, hidden)
        # query = self.dropout(x)
        # attn_output, alpha_n = self.attention_net(x, query)
        # print(attn_output.size()) # size hidden*2 *1 union

        x = self.fc(attn_output.squeeze(0)) #Linear

        # s,b,h = x.size()
#         print("{} {} {}".format(s,b,h))
        # x = x.reshape(s,b*h)
        # x = self.fc1(x)
        # print(x)
        return x

max_len = 512
batchsize = 512 #512
kind_size = 10
epoch_num = 100

data_save_dir = '../data_dir'
model_save_dir = '../models'

result_save_dir = '../result_dir'

data_use_name = ''
vul_use_name = ''

def data_making_all():
    print("{} processing!".format(vul_use_name))

    files_bytecodes = None
    with open('...') as fp:
        files_bytecodes = json.load(fp)
        
    reentrancy_files = None
    with open('...' + data_use_name + '.json') as fp:
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

def data_making():
    print("{} processing!".format(vul_use_name))
    files_bytecodes = None
    with open('...') as fp:
        files_bytecodes = json.load(fp)
        
    reentrancy_files = None
    with open('...' + data_use_name + '.json') as fp:
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
    criterion = torch.nn.CrossEntropyLoss().cuda()
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
            inputs = inputs.cuda().float()
            target = target.cuda().long()
            optimizer.zero_grad()  
            outputs = model(inputs)   
            # print(type(outputs))
            bias = torch.sub(outputs.data[:,0], outputs.data[:,1], alpha=1)
            bag_index = {}
            bags = bags.numpy()
            for i in range(len(bags)):
                if bags[i] not in bag_index:
                    bag_index[bags[i]] = []
                bag_index[bags[i]].append(i)
            
            ...

    def updata_data(dataloader):
        print("update data")
        # print("lr: {}".format(scheduler.get_lr()))
    #     print(type(DataLoader_train))
        datasets_data = []
        class_bias = []
        datasets_labels = []
        datasets_bags = []

        with torch.no_grad():   
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                datasets_data.extend(inputs.numpy())
                datasets_labels.extend(labels.numpy())
                datasets_bags.extend(bags.numpy())

                inputs = inputs.cuda().float()
                labels = labels.cuda().long()
                ...
    
    def test(dataloader):
        # correct = 0
        # total = 0
        print("test datasets")
        true_labels = []
        predict_labels = []
        class_bags = []
        with torch.no_grad():   
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                inputs = inputs.cuda().float()
                ...

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
        ...

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
        with torch.no_grad():   
            for batch_idx,data in enumerate(dataloader):
                inputs,labels,bags = data
                inputs = inputs.cuda().float()
                # labels = labels.cuda().long()
                labels = list(labels)
                
                ...
                
    data_json = json.dumps(file_label, cls=NpEncoder)
    fileObject = open(os.path.join(result_save_dir, vul_use_name + '_result.json'), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()
    print("{} save successfully!".format(vul_use_name + '_result.json'))

    # del true_labels, predict_labels
    del true_labels, predict_labels, instances_predict_labels, model, file_label
    gc.collect()
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

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

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
                    data_val = np.zeros(data_len)
                    for j in range(min(data_len, len(data_inputs[idx[i]][k]))):
                        data_val[j] = data_inputs[idx[i]][k][j]
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
                for k in range(min(kind_size, len(data_inputs[i]))):
                    data_val = np.zeros(data_len)
                    for j in range(min(data_len, len(data_inputs[i][k]))):
                        data_val[j] = data_inputs[i][k][j]
                    self.data.append(data_val)
                    self.label.append(label_inputs[i])
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
                    data_val = np.zeros(data_len)
                    for j in range(min(data_len, len(data_inputs[idx[i]][k]))):
                        data_val[j] = data_inputs[idx[i]][k][j]
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
                    data_val = np.zeros(data_len)
                    for j in range(min(data_len, len(data_inputs[i][k]))):
                        data_val[j] = data_inputs[i][k][j]
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

    def __len__(self):
        return len(self.data)

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

    model = RandomForestClassifier(n_estimators=50, class_weight='balanced')

    def train(epoch, dataloader):
        print("train datasets")
        inputs = dataloader.data
        target = dataloader.label
        bags = dataloader.bag

        model.fit(inputs, target)

    def updata_data(dataloader):
        print("update data")

        datasets_data = dataloader.data
        datasets_labels = dataloader.label
        datasets_bags = dataloader.bag

        outputs = model.predict_proba(datasets_data)
        class_bias = outputs[:,0] - outputs[:,1]

        print("end of predict")

        df = pd.DataFrame({'bags': datasets_bags, 'labels': datasets_labels, 'bias': class_bias})
        group = df.groupby('bags')
        indexlist = []
        for bag,x in group:
            if(x['labels'].values[0]==0):
                indexlist.extend(x.index.tolist())
            else:
                indexlist.extend(x['bias'].sort_values(ascending=False, inplace=False).index.tolist()[int((len(x)+1)*0.8):])
                # indexlist.append(x['bias'].idxmin())
        del df, group
        gc.collect()
        print("data length: {}".format(len(indexlist)))
        return datasets_data[indexlist], datasets_labels[indexlist], datasets_bags[indexlist]
    
    def test(dataloader):
        
        print("test datasets")

        inputs = dataloader.data
        true_labels = dataloader.label
        class_bags = dataloader.bag

        predict_labels = model.predict(inputs)
        
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
        if epoch == 0:
            train(epoch,trainset)
        else:
            train(epoch,DataLoader_train_upd)
        
        new_datas, new_labels, new_bags = updata_data(trainset)
        print("ceshi!")
        DataLoader_train_upd = Dataset_Maker_Mulins_Upd(new_datas, new_labels, new_bags)
        
        del new_datas, new_labels
        gc.collect()

        true_labels, predict_labels = test(testset)

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
            joblib.dump(model, os.path.join(model_save_dir, "RF_" + vul_use_name + ".pkl"))
            print("model save {}!".format(model_save_dir))
        print("Best accuracy on test set, epoch: {}, accuracy: {}, recall: {}, precision: {}, f1: {}".format(best_epoch, best_acc, best_recall, best_precision, best_f1))

        del true_labels, predict_labels
        gc.collect()

def test_LSTM_with_multi_instance(test_bytecodes, test_labels):
    testset = Dataset_Maker_Mulins('Test', test_bytecodes, test_labels, max_len, kind_size)

    model = joblib.load(os.path.join(model_save_dir, "RF_" + vul_use_name + ".pkl"))

    def test(dataloader):
        
        print("test datasets")

        inputs = dataloader.data
        true_labels = dataloader.label
        class_bags = dataloader.bag

        predict_labels = model.predict(inputs)
        
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

        del df, group
        gc.collect()
        # print(predict_labels)
        return true_labels, predict_labels
    
    true_labels, predict_labels = test(testset)
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
    
    model = joblib.load(os.path.join(model_save_dir, "RF_" + vul_use_name + ".pkl"))

    def test(dataloader):
        
        print("test datasets")

        inputs = dataloader.data
        true_labels = dataloader.label
        class_bags = dataloader.bag

        predict_labels = model.predict(inputs)
        
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
        return true_labels, predict_labels, instances_predict_labels
    
    file_label = {}
    true_labels, predict_labels, instances_predict_labels = test(testset)
    # print(predict_labels[:50])
    for i in range(len(true_labels)):
        file_label[true_labels[i]] = (predict_labels[i], instances_predict_labels[i])

    # print("Accuracy on test set: {}, recall: {}, precision: {}, f1: {}".format(acc, recall, precision, F1))
    data_json = json.dumps(file_label, cls=NpEncoder)
    fileObject = open(os.path.join(result_save_dir, vul_use_name + '_result.json'), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()
    print("{} save successfully!".format(vul_use_name + '_result.json'))

    # del true_labels, predict_labels
    del true_labels, predict_labels, instances_predict_labels, model, file_label
    gc.collect()

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

    predict_results = {}

    for vul_use_name in vul_use_names:
        print(vul_use_name + ' precessing!')
        
        model = joblib.load(os.path.join(model_save_dir, "RF_" + vul_use_name + ".pkl"))

        def test(dataloader):
            # print("test datasets")
            inputs = dataloader.data
            true_labels = dataloader.label
            class_bags = dataloader.bag

            predict_labels = model.predict(inputs)
            
            df = pd.DataFrame({'bags': class_bags, 'labels': true_labels, 'prelabels': predict_labels})
            group = df.groupby('bags')
            true_labels = []
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
            return predict_labels, instances_predict_labels
    
        # true_labels, predict_labels = test(DataLoader_test)
        predict_labels, instances_predict_labels = test(testset)

        predict_results[vul_use_name] = {'predict_labels': predict_labels, 'instances_predict_labels':instances_predict_labels}

        del predict_labels, model, instances_predict_labels
        gc.collect()

    return predict_results
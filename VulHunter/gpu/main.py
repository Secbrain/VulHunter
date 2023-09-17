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

import result_predict
import bytecodes_construction

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

if __name__ == '__main__':
    # global vul_use_name

    # vul_use_name = sys.argv[1]
    # vul_use_name = timestamp #shadowing-state

    filepath = './BitsForAI.sol'
    version = '0.5.16'

    time_begin = time.time()

    bytecodes_lists = []
    bytecodes_list_val = bytecodes_construction.make_bytecodes_for_solidity_file(filepath, version)
    bytecodes_lists.append(bytecodes_list_val)

    time_instance = time.time()

    vul_use_names = ['reentrancy', 'controlled-array-length', 'suicidal', 'controlled-delegatecall', 'arbitrary-send', 'incorrect-equality', 'integer-overflow', 'unchecked-lowlevel', 'tx-origin', 'locked-ether', 'unchecked-send', 'costly-loop', 'erc721-interface', 'erc20-interface', 'timestamp', 'block-other-parameters', 'calls-loop', 'low-level-calls', 'erc20-indexed', 'erc20-throw', 'hardcoded', 'array-instead-bytes', 'unused-state', 'costly-operations-loop', 'external-function', 'send-transfer', 'boolean-equal', 'boolean-cst', 'uninitialized-state', 'tod']

    predict_results = result_predict.test_LSTM_prelabel(bytecodes_lists, vul_use_names)

    time_predict = time.time()

    print(predict_results)

    print("time overhead: {} {}, all: {}".format(time_instance-time_begin, time_predict-time_instance, time_predict-time_begin))
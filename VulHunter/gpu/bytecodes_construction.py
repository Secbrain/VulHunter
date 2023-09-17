# -*- coding: utf-8 -*-
# import torch.utils.data as data
# from PIL import Image
import numpy as np
import shutil
import errno
# import torch
import os
import io
# from tqdm import tqdm
import numpy as np
# import torch
import re
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim

import sys
# from sklearn import metrics
# import sklearn
import json

from decimal import *
from pyevmasm import disassemble_hex
from pyevmasm import assemble_hex

# from solc import compile_source, compile_files, link_code
from solcx import compile_source, compile_files, link_code, get_installed_solc_versions, set_solc_version
from evm_cfg_builder.cfg import CFG

# import joblib
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
# import time
# import random
import gc

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

def cycle_research(current_opcodes_list, current_bytecodes_list, current_blocks_list, current_block, cycleblock_num, evm_bytecodes_maps, all_evm_bytecodes_lists):
    link_block = '@{:x}-{:x}'.format(current_block.start.pc, current_block.end.pc)
    if link_block in current_blocks_list:
        cycleblock_num += 1

    current_blocks_list.append(link_block)
#     current_opcodes_list.extend(evm_opcodes_maps[link_block])
    current_bytecodes_list.extend(evm_bytecodes_maps[link_block])

    if cycleblock_num == 2 or len(current_blocks_list) >= 32:
        ...
                
    return all_evm_bytecodes_lists

def cycle_research_continue2(current_opcodes_list, current_bytecodes_list, current_blocks_list, current_block, cycleblock_num, evm_bytecodes_maps, all_evm_bytecodes_lists):
    link_block = '@{:x}-{:x}'.format(current_block.start.pc, current_block.end.pc)
    if link_block in current_blocks_list:
        cycleblock_num += 1
    else:
        cycleblock_num = 0

    current_blocks_list.append(link_block)
#     current_opcodes_list.extend(evm_opcodes_maps[link_block])
    current_bytecodes_list.extend(evm_bytecodes_maps[link_block])

    if cycleblock_num == 2 or len(current_blocks_list) >= 32:
        ...
    else:
        ...
                
    return all_evm_bytecodes_lists

def make_bytecodes():
    file_solc_versions = None
    with open('...') as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes = {}
    files_bytecodes_without_data = {}
    root_dir = "..."
    index = 0
    compile_true = 0
    for contract_dir_name in os.listdir(root_dir):
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(file_path) + "!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

                evm_bytecodes_maps = {}

                result = compile_source(file_content)
                begin_blocks = []
                for k,v in result.items():

                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    ...
                ...
                
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                # print(len(bytecodes_lists[0]))
                files_bytecodes[contract_file_name] = bytecodes_lists[:10] #10
                compile_true += 1
                del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                gc.collect()
            except Exception:
    #             print("except:")
                try:
                    set_solc_version('v0.4.24')
                    f = open(contract_file_path)
                    file_content = f.read()
                    f.close()

                    evm_bytecodes_maps = {}

                    ...
                except Exception as e:
                    # print(e)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("...", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_avgloc():
    file_solc_versions = None
    ...

def make_bytecodes_continue2_bytecode():
    codes_path = r"hashcode"
    ...

def make_bytecodes_continue2():
    file_solc_versions = None
    ...
    
def make_bytecodes_continue2_multi_methods():
    file_solc_versions = None
    ...

def make_bytecodes_continue2_avg():
    file_solc_versions = None
    ...

def make_bytecodes_continue2_bigdataset_1():
    path = './etherscan_5006'
    inpath = './contract_informations_Deduplication_json_etherscan_5006.csv'
    contracts_colname=['address','compileversion','name','useversion'];
    csvfile = io.open(inpath,'r',encoding="utf-8")
    df = pd.read_csv(csvfile,header=None,names=contracts_colname)
    csvfile.close()
    
    files_bytecodes = {}

    ...

def make_bytecodes_continue2_event():
    path = './eventcontracts'
    inpath = './contract_informations_Deduplication_json_events.csv'
    contracts_colname=['address','compileversion','name','useversion'];
    csvfile = io.open(inpath,'r',encoding="utf-8")
    df = pd.read_csv(csvfile,header=None,names=contracts_colname)
    csvfile.close()
    
    files_bytecodes = {}

    compile_true = 0
    false_files = []

    ...

def make_bytecodes_for_solidity_file(file_path, solc_version):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            ...
    return None

def make_bytecodes_for_binary_file(file_path):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()

        code = file_content[0].strip()
    
        evm_bytecodes_maps = {}
        begin_blocks = []
        
        cfg = CFG(code)
#                 print(len(cfg.basic_blocks))
        ...
    except Exception as e:
        print("construct failed!")
        return None

# if __name__ == '__main__':
#     # make_bytecodes()
#     # make_bytecodes_avgloc()
#     # make_bytecodes_continue2_bytecode()
#     # make_bytecodes_continue2()
#     # make_bytecodes_continue2_bigdataset_1()
#     make_bytecodes_continue2_event()
#     # make_bytecodes_continue2_avg()
#     # make_bytecodes_continue2_multi_methods()
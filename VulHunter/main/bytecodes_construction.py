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
import pyevmasm
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
#         all_evm_opcodes_lists.append(current_opcodes_list)
        all_evm_bytecodes_lists.append(current_bytecodes_list)
#         all_blocks_name.append(current_blocks_list)
    else:
        if len(current_block.all_outgoing_basic_blocks) == 0:
#             all_evm_opcodes_lists.append(current_opcodes_list)
            all_evm_bytecodes_lists.append(current_bytecodes_list)
#             all_blocks_name.append(current_blocks_list)
        else:
            for i in range(len(current_block.all_outgoing_basic_blocks)):
                all_evm_bytecodes_lists = cycle_research(current_opcodes_list.copy(), current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, evm_bytecodes_maps, all_evm_bytecodes_lists)
                
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
#         all_evm_opcodes_lists.append(current_opcodes_list)
        all_evm_bytecodes_lists.append(current_bytecodes_list)
#         all_blocks_name.append(current_blocks_list)
    else:
        if len(current_block.all_outgoing_basic_blocks) == 0:
#             all_evm_opcodes_lists.append(current_opcodes_list)
            all_evm_bytecodes_lists.append(current_bytecodes_list)
#             all_blocks_name.append(current_blocks_list)
        else:
            for i in range(len(current_block.all_outgoing_basic_blocks)):
                all_evm_bytecodes_lists = cycle_research_continue2(current_opcodes_list.copy(), current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, evm_bytecodes_maps, all_evm_bytecodes_lists)
                
    return all_evm_bytecodes_lists

def cycle_research_continue2_op_and_byte(current_opcodes_list, current_bytecodes_list, current_blocks_list, current_block, cycleblock_num, evm_bytecodes_maps, evm_opcodes_maps, all_evm_bytecodes_lists, all_evm_opcodes_lists):
    link_block = '@{:x}-{:x}'.format(current_block.start.pc, current_block.end.pc)
    if link_block in current_blocks_list:
        cycleblock_num += 1
    else:
        cycleblock_num = 0

    current_blocks_list.append(link_block)
    current_opcodes_list.extend(evm_opcodes_maps[link_block])
    current_bytecodes_list.extend(evm_bytecodes_maps[link_block])

    if cycleblock_num == 2 or len(current_blocks_list) >= 32:
        all_evm_opcodes_lists.append(current_opcodes_list)
        all_evm_bytecodes_lists.append(current_bytecodes_list)
#         all_blocks_name.append(current_blocks_list)
    else:
        if len(current_block.all_outgoing_basic_blocks) == 0:
            all_evm_opcodes_lists.append(current_opcodes_list)
            all_evm_bytecodes_lists.append(current_bytecodes_list)
#             all_blocks_name.append(current_blocks_list)
        else:
            for i in range(len(current_block.all_outgoing_basic_blocks)):
                all_evm_bytecodes_lists, all_evm_opcodes_lists = cycle_research_continue2_op_and_byte(current_opcodes_list.copy(), current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, evm_bytecodes_maps, evm_opcodes_maps, all_evm_bytecodes_lists, all_evm_opcodes_lists)
                
    return all_evm_bytecodes_lists, all_evm_opcodes_lists

def cycle_research_continue2_op_and_byte_and_pc(current_opcodes_list, current_bytecodes_list, current_pc_list, current_blocks_list, current_block, cycleblock_num, evm_bytecodes_maps, evm_opcodes_maps, evm_pc_maps, all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists):
    link_block = '@{:x}-{:x}'.format(current_block.start.pc, current_block.end.pc)
    if link_block in current_blocks_list:
        cycleblock_num += 1
    else:
        cycleblock_num = 0

    current_blocks_list.append(link_block)
    current_opcodes_list.extend(evm_opcodes_maps[link_block])
    current_bytecodes_list.extend(evm_bytecodes_maps[link_block])
    current_pc_list.extend(evm_pc_maps[link_block])

    if cycleblock_num == 2 or len(current_blocks_list) >= 32:
        all_evm_opcodes_lists.append(current_opcodes_list)
        all_evm_bytecodes_lists.append(current_bytecodes_list)
        all_evm_pc_lists.append(current_pc_list)
#         all_blocks_name.append(current_blocks_list)
    else:
        if len(current_block.all_outgoing_basic_blocks) == 0:
            all_evm_opcodes_lists.append(current_opcodes_list)
            all_evm_bytecodes_lists.append(current_bytecodes_list)
            all_evm_pc_lists.append(current_pc_list)
#             all_blocks_name.append(current_blocks_list)
        else:
            for i in range(len(current_block.all_outgoing_basic_blocks)):
                all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists = cycle_research_continue2_op_and_byte_and_pc(current_opcodes_list.copy(), current_bytecodes_list.copy(), current_pc_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, evm_bytecodes_maps, evm_opcodes_maps, evm_pc_maps, all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists)
                
    return all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists

def make_bytecodes():
    file_solc_versions = None
    with open('/mnt/docker/solcversions.json') as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes = {}
    files_bytecodes_without_data = {}
    root_dir = "/mnt/docker/contract"
    index = 0
    compile_true = 0
    for contract_dir_name in os.listdir(root_dir):
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(contract_file_path) + "is not a solidity file!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

        #             evm_opcodes_maps = {}
                evm_bytecodes_maps = {}

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)
                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
    #             print(len(begin_blocks))
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

        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                    result = compile_source(file_content)

                    begin_blocks = []
                    for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                        if len(v['bin-runtime']) == 0:
                            continue
                        cfg = CFG(v['bin-runtime'])
            #                 print(len(cfg.basic_blocks))
                        for i in range(len(cfg.basic_blocks)):
                            basic_block = cfg.basic_blocks[i]
                            if len(basic_block.all_incoming_basic_blocks) == 0:
                                begin_blocks.append(basic_block)
            #                     evm_opcodes_list = []
                            evm_bytecodes_list = []
                            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                            #                                           basic_block.end.pc))
                            #     print('\t\tInstructions:')
                            for ins in basic_block.instructions:
                #                 evm_opcodes_list.append(ins.name) #opcode
                                # evm_bytecodes_list.append(ins.name) #bytecode
                                evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                                # if ins.operand is not None: #values list
                                #     evm_bytecodes_list.append(ins.operand)
                            evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        del cfg;
                        gc.collect()


                    #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
            #         all_evm_opcodes_lists = []
                    all_evm_bytecodes_lists = []
            #         all_blocks_name = []

        #                     max_num = 0
                    for begin_block in begin_blocks:
                        all_evm_bytecodes_lists.extend(cycle_research([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                    # print(len(bytecodes_lists[0]))
                    files_bytecodes[contract_file_name] = bytecodes_lists[:10]
                    compile_true += 1
                    del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                    gc.collect()
                except Exception as e:
                    # print(e)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("/mnt/docker/contract_bytecode3_list10.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_avgloc():
    file_solc_versions = None
    with open('/mnt/docker/solcversions.json') as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes = {}
    files_bytecodes_without_data = {}
    root_dir = "/mnt/docker/contract"
    index = 0
    compile_true = 0
    for contract_dir_name in os.listdir(root_dir):
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(contract_file_path) + "is not a solidity file!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

        #             evm_opcodes_maps = {}
                evm_bytecodes_maps = {}

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)
                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
    #             print(len(begin_blocks))
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                # print(len(bytecodes_lists[0]))

                locs = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)

                bytecodes_val = []
                for loc in locs:
                    bytecodes_val.append(bytecodes_lists[loc])

                files_bytecodes[contract_file_name] = bytecodes_val

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

        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                    result = compile_source(file_content)

                    begin_blocks = []
                    for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                        if len(v['bin-runtime']) == 0:
                            continue
                        cfg = CFG(v['bin-runtime'])
            #                 print(len(cfg.basic_blocks))
                        for i in range(len(cfg.basic_blocks)):
                            basic_block = cfg.basic_blocks[i]
                            if len(basic_block.all_incoming_basic_blocks) == 0:
                                begin_blocks.append(basic_block)
            #                     evm_opcodes_list = []
                            evm_bytecodes_list = []
                            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                            #                                           basic_block.end.pc))
                            #     print('\t\tInstructions:')
                            for ins in basic_block.instructions:
                #                 evm_opcodes_list.append(ins.name) #opcode
                                # evm_bytecodes_list.append(ins.name) #bytecode
                                evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                                # if ins.operand is not None: #values list
                                #     evm_bytecodes_list.append(ins.operand)
                            evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        del cfg;
                        gc.collect()


                    #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
            #         all_evm_opcodes_lists = []
                    all_evm_bytecodes_lists = []
            #         all_blocks_name = []

        #                     max_num = 0
                    for begin_block in begin_blocks:
                        all_evm_bytecodes_lists.extend(cycle_research([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                    locs = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)

                    bytecodes_val = []
                    for loc in locs:
                        bytecodes_val.append(bytecodes_lists[loc])

                    files_bytecodes[contract_file_name] = bytecodes_val
                    
                    compile_true += 1
                    del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                    gc.collect()
                except Exception as e:
                    # print(e)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("/mnt/docker/contract_bytecode3_list10_avg.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_continue2_bytecode():
    codes_path = r"hashcode"
    codes = None
    with open(codes_path,'r') as f:
        codes = f.readlines()
    
    compile_true = 0
    files_bytecodes = {}
    
    for i in range(160000, len(codes)):
        # print(i)
        code_index = codes[i].strip()[:codes[i].find('#')]
        code = codes[i].strip()[codes[i].find('#')+1:]
    
        try:
            evm_bytecodes_maps = {}
            begin_blocks = []
            
            cfg = CFG(code)
    #                 print(len(cfg.basic_blocks))
            for ii in range(len(cfg.basic_blocks)):
                basic_block = cfg.basic_blocks[ii]
                if len(basic_block.all_incoming_basic_blocks) == 0:
                    begin_blocks.append(basic_block)
    #                     evm_opcodes_list = []
                evm_bytecodes_list = []
                link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                #                                           basic_block.end.pc))
                #     print('\t\tInstructions:')
                for ins in basic_block.instructions:
    #                 evm_opcodes_list.append(ins.name) #opcode
                    # evm_bytecodes_list.append(ins.name) #bytecode
                    evm_bytecodes_list.append(ins.opcode) #bytecode
    #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                    # if ins.operand is not None: #values
                    #     evm_bytecodes_list.append(ins.operand)
                evm_bytecodes_maps[link_block] = evm_bytecodes_list
            
            del cfg;
            gc.collect()

            all_evm_bytecodes_lists = []

            for begin_block in begin_blocks:
                all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
            bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
            # print(len(bytecodes_lists[0]))
            files_bytecodes[code_index] = bytecodes_lists[:10] #10
            compile_true += 1
            del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists
            gc.collect()
        except Exception as e:
            print(e)
            print("{} {} compiled faild!".format(i, code_index))
        if (i+1) % 400 == 0:
            print("current file index: {}, success: {}".format(i, compile_true))
        if (i+1) % 20000 == 0:
            data_json = json.dumps(files_bytecodes, cls=NpEncoder)
            fileObject = open("contract_bytecode3_list10_continue2_{}.json".format(i+1), 'w') #contract_bytecodes
            fileObject.write(data_json)
            fileObject.close()
            print("file save: {}, success: {}".format(i+1, compile_true))
            files_bytecodes = {}
            del data_json, fileObject
            gc.collect()
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("contract_bytecode3_list10_continue2_{}.json".format(i+1), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()
    print("file save: {}, success: {}".format(i+1, compile_true))

def make_bytecodes_continue2(contract_root_dir, solcversions_path, instance_len, instance_dir):
    file_solc_versions = None
    with open(solcversions_path) as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes = {}
    root_dir = contract_root_dir
    index = 0
    compile_true = 0
    false_files = []
    file_dir_names = list(os.listdir(root_dir))
    file_dir_names.sort()
    for contract_dir_name in file_dir_names:
        print(contract_dir_name)
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(contract_file_path) + "is not a solidity file!")
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
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)

                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

    #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
    #             print(len(begin_blocks))
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                # print(len(bytecodes_lists[0]))
                files_bytecodes[contract_file_name] = bytecodes_lists[:instance_len] #10
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

        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                    result = compile_source(file_content)

                    begin_blocks = []
                    for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                        if len(v['bin-runtime']) == 0:
                            continue
                        cfg = CFG(v['bin-runtime'])
            #                 print(len(cfg.basic_blocks))
                        for i in range(len(cfg.basic_blocks)):
                            basic_block = cfg.basic_blocks[i]
                            if len(basic_block.all_incoming_basic_blocks) == 0:
                                begin_blocks.append(basic_block)
            #                     evm_opcodes_list = []
                            evm_bytecodes_list = []
                            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                            #                                           basic_block.end.pc))
                            #     print('\t\tInstructions:')
                            for ins in basic_block.instructions:
                #                 evm_opcodes_list.append(ins.name) #opcode
                                # evm_bytecodes_list.append(ins.name) #bytecode
                                evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                                # if ins.operand is not None: #values list
                                #     evm_bytecodes_list.append(ins.operand)
                            evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        del cfg;
                        gc.collect()


                    #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
            #         all_evm_opcodes_lists = []
                    all_evm_bytecodes_lists = []
            #         all_blocks_name = []

        #                     max_num = 0
                    for begin_block in begin_blocks:
                        all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                    # print(len(bytecodes_lists[0]))
                    files_bytecodes[contract_file_name] = bytecodes_lists[:instance_len]
                    compile_true += 1
                    del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                    gc.collect()
                except Exception as e:
                    # print(e)
                    false_files.append(contract_file_name)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open(os.path.join(instance_dir, "contract_bytecodes_list10.json"), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(false_files, cls=NpEncoder)
    fileObject = open(os.path.join(instance_dir, "contract_falsecompile_list.json"), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()
    
def make_bytecodes_continue2_multi_methods():
    file_solc_versions = None
    with open('/mnt/docker/solcversionsnew.json') as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes_min = {}
    files_bytecodes_avg = {}
    files_bytecodes_random = {}
    root_dir = "/mnt/docker/contract"
    index = 0
    compile_true = 0
    false_files = []
    file_dir_names = list(os.listdir(root_dir))
    file_dir_names.sort()
    for contract_dir_name in file_dir_names:
        print(contract_dir_name)
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(contract_file_path) + "is not a solidity file!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

        #             evm_opcodes_maps = {}
                evm_bytecodes_maps = {}

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)
                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
    #             print(len(begin_blocks))
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = False)
                # print(len(bytecodes_lists[0]))
                files_bytecodes_min[contract_file_name] = bytecodes_lists[:10] #10

                if len(bytecodes_lists) < 10:
                    files_bytecodes_avg[contract_file_name] = bytecodes_lists
                    files_bytecodes_random[contract_file_name] = bytecodes_lists
                else:
                    locs_avg = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)
                    locs_random = np.random.choice(len(bytecodes_lists), 10, replace=False)
                    bytecodes_val_avg = []
                    bytecodes_val_random = []
                    for loc_val in range(len(locs_avg)):
                        bytecodes_val_avg.append(bytecodes_lists[locs_avg[loc_val]])
                        bytecodes_val_random.append(bytecodes_lists[locs_random[loc_val]])

                    files_bytecodes_avg[contract_file_name] = bytecodes_val_avg
                    files_bytecodes_random[contract_file_name] = bytecodes_val_random

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

        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                    result = compile_source(file_content)

                    begin_blocks = []
                    for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                        if len(v['bin-runtime']) == 0:
                            continue
                        cfg = CFG(v['bin-runtime'])
            #                 print(len(cfg.basic_blocks))
                        for i in range(len(cfg.basic_blocks)):
                            basic_block = cfg.basic_blocks[i]
                            if len(basic_block.all_incoming_basic_blocks) == 0:
                                begin_blocks.append(basic_block)
            #                     evm_opcodes_list = []
                            evm_bytecodes_list = []
                            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                            #                                           basic_block.end.pc))
                            #     print('\t\tInstructions:')
                            for ins in basic_block.instructions:
                #                 evm_opcodes_list.append(ins.name) #opcode
                                # evm_bytecodes_list.append(ins.name) #bytecode
                                evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                                # if ins.operand is not None: #values list
                                #     evm_bytecodes_list.append(ins.operand)
                            evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        del cfg;
                        gc.collect()

        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
            #         all_evm_opcodes_lists = []
                    all_evm_bytecodes_lists = []
            #         all_blocks_name = []

        #                     max_num = 0
                    for begin_block in begin_blocks:
                        all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = False)
                    # print(len(bytecodes_lists[0]))
                    files_bytecodes_min[contract_file_name] = bytecodes_lists[:10] #10

                    if len(bytecodes_lists) < 10:
                        files_bytecodes_avg[contract_file_name] = bytecodes_lists
                        files_bytecodes_random[contract_file_name] = bytecodes_lists
                    else:
                        locs_avg = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)
                        locs_random = np.random.choice(len(bytecodes_lists), 10, replace=False)
                        bytecodes_val_avg = []
                        bytecodes_val_random = []
                        for loc_val in range(len(locs_avg)):
                            bytecodes_val_avg.append(bytecodes_lists[locs_avg[loc_val]])
                            bytecodes_val_random.append(bytecodes_lists[locs_random[loc_val]])

                        files_bytecodes_avg[contract_file_name] = bytecodes_val_avg
                        files_bytecodes_random[contract_file_name] = bytecodes_val_random

                    compile_true += 1
                    del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                    gc.collect()
                except Exception as e:
                    # print(e)
                    false_files.append(contract_file_name)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes_avg, cls=NpEncoder)
    fileObject = open("/mnt/wangxinban/contracts/contract_bytecodes_list10_avgloc.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(files_bytecodes_min, cls=NpEncoder)
    fileObject = open("/mnt/wangxinban/contracts/contract_bytecodes_list10_minloc.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(files_bytecodes_random, cls=NpEncoder)
    fileObject = open("/mnt/wangxinban/contracts/contract_bytecodes_list10_randomloc.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(false_files, cls=NpEncoder)
    fileObject = open("/mnt/wangxinban/contracts/contract_falsecompile_list_three_types.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_continue2_avg():
    file_solc_versions = None
    with open('/mnt/docker/solcversionsnew.json') as fp:
        file_solc_versions = json.load(fp)

    files_bytecodes = {}
    root_dir = "/mnt/docker/contract"
    index = 0
    compile_true = 0
    false_files = []
    file_dir_names = list(os.listdir(root_dir))
    file_dir_names.sort()
    for contract_dir_name in file_dir_names:
        print(contract_dir_name)
        contract_dir = os.path.join(root_dir, contract_dir_name)
        for contract_file_name in os.listdir(contract_dir):
            try:
                contract_file_path = os.path.join(contract_dir, contract_file_name)
                if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                    print(str(contract_file_path) + "is not a solidity file!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

        #             evm_opcodes_maps = {}
                evm_bytecodes_maps = {}

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)
                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
    #             print(len(begin_blocks))
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                # print(len(bytecodes_lists[0]))

                if len(bytecodes_lists) < 10:
                    files_bytecodes[contract_file_name] = bytecodes_lists
                else:
                    locs = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)

                    bytecodes_val = []
                    for loc in locs:
                        bytecodes_val.append(bytecodes_lists[loc])

                    files_bytecodes[contract_file_name] = bytecodes_val

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

        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                    result = compile_source(file_content)

                    begin_blocks = []
                    for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                        if len(v['bin-runtime']) == 0:
                            continue
                        cfg = CFG(v['bin-runtime'])
            #                 print(len(cfg.basic_blocks))
                        for i in range(len(cfg.basic_blocks)):
                            basic_block = cfg.basic_blocks[i]
                            if len(basic_block.all_incoming_basic_blocks) == 0:
                                begin_blocks.append(basic_block)
            #                     evm_opcodes_list = []
                            evm_bytecodes_list = []
                            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                            #                                           basic_block.end.pc))
                            #     print('\t\tInstructions:')
                            for ins in basic_block.instructions:
                #                 evm_opcodes_list.append(ins.name) #opcode
                                # evm_bytecodes_list.append(ins.name) #bytecode
                                evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                                # if ins.operand is not None: #values list
                                #     evm_bytecodes_list.append(ins.operand)
                            evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        del cfg;
                        gc.collect()


        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
            #         all_evm_opcodes_lists = []
                    all_evm_bytecodes_lists = []
            #         all_blocks_name = []

        #                     max_num = 0
                    for begin_block in begin_blocks:
                        all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                # print(len(bytecodes_lists[0]))

                    if len(bytecodes_lists) < 10:
                        files_bytecodes[contract_file_name] = bytecodes_lists
                    else:
                        locs = np.linspace(0, len(bytecodes_lists)-1, 10, dtype=int)

                        bytecodes_val = []
                        for loc in locs:
                            bytecodes_val.append(bytecodes_lists[loc])

                        files_bytecodes[contract_file_name] = bytecodes_val

                    compile_true += 1
                    del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                    gc.collect()
                except Exception as e:
                    # print(e)
                    false_files.append(contract_file_name)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("/mnt/wangxinban/contracts/contract_bytecodes_list10_avg.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    # data_json = json.dumps(false_files, cls=NpEncoder)
    # fileObject = open("/mnt/wangxinban/contracts/contract_falsecompile_list.json", 'w') #contract_bytecodes
    # fileObject.write(data_json)
    # fileObject.close()

def make_bytecodes_continue2_bigdataset_1():
    path = './etherscan_5006'
    inpath = './contract_informations_Deduplication_json_etherscan_5006.csv'
    contracts_colname=['address','compileversion','name','useversion'];
    csvfile = io.open(inpath,'r',encoding="utf-8")
    df = pd.read_csv(csvfile,header=None,names=contracts_colname)
    csvfile.close()
    
    files_bytecodes = {}

    compile_true = 0
    false_files = []

    for index,row in df.iterrows():
        file = row['name']
        try:
            contract_file_path = os.path.join(path, file)
            if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                print(str(contract_file_path) + "is not a solidity file or file is not exist!")
                continue

            set_solc_version('v' + row['useversion'])

            f = open(contract_file_path)
            file_content = f.read()
            f.close()

    #             evm_opcodes_maps = {}
            evm_bytecodes_maps = {}

#             set_solc_version('v0.4.24')
#             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
            result = compile_source(file_content)
            begin_blocks = []
            for k,v in result.items():
    #             print(len(v['bin-runtime']))
    #             print(v['bin-runtime'])
    #             print(v)
                if len(v['bin-runtime']) == 0:
                    continue
                cfg = CFG(v['bin-runtime'])
    #                 print(len(cfg.basic_blocks))
                for i in range(len(cfg.basic_blocks)):
                    basic_block = cfg.basic_blocks[i]
                    if len(basic_block.all_incoming_basic_blocks) == 0:
                        begin_blocks.append(basic_block)
    #                     evm_opcodes_list = []
                    evm_bytecodes_list = []
                    link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                    #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                    #                                           basic_block.end.pc))
                    #     print('\t\tInstructions:')
                    for ins in basic_block.instructions:
        #                 evm_opcodes_list.append(ins.name) #opcode
                        # evm_bytecodes_list.append(ins.name) #bytecode
                        evm_bytecodes_list.append(ins.opcode) #bytecode
    #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                        # if ins.operand is not None: #values
                        #     evm_bytecodes_list.append(ins.operand)
                    evm_bytecodes_maps[link_block] = evm_bytecodes_list
                del cfg;
                gc.collect()

    #             file_bytecodes = []
    #             for contract_index in result:
    #                 file_bytecodes.append(result[contract_index]['opcodes'])
    #             file_bytecodes_str = ' '.join(file_opcodes)

            #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
#             print(len(list(evm_bytecodes_maps.keys())))
    #                 print(evm_bytecodes_maps.keys())
    #         all_evm_opcodes_lists = []
            all_evm_bytecodes_lists = []
    #         all_blocks_name = []
    #                 max_num = 0           
#             print(len(begin_blocks))
            for begin_block in begin_blocks:
                all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

#             print(all_evm_bytecodes_lists)
            bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
            # print(len(bytecodes_lists[0]))
            files_bytecodes[file] = bytecodes_lists[:10] #10
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

    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)

                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values list
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()


                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #                     print(len(list(evm_bytecodes_maps.keys())))
    #                     print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []

    #                     max_num = 0
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                # print(len(bytecodes_lists[0]))
                files_bytecodes[file] = bytecodes_lists[:10]
                compile_true += 1
                del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                gc.collect()
            except Exception as e:
                # print(e)
                false_files.append(file)
                print(file + " compiled faild!")
        # print(index)
        if (index+1) % 300 == 0:
            print("current file index: {}, success: {}".format(index+1, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("./contract_bytecodes_list10_bigdataset1_5006.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(false_files, cls=NpEncoder)
    fileObject = open("./contract_falsecompile_list_bigdataset1_5006.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

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

    for index,row in df.iterrows():
        file = row['name']
        try:
            contract_file_path = os.path.join(path, file)
            if not (os.path.isfile(contract_file_path) and os.path.splitext(contract_file_path)[1]=='.sol'):
                print(str(contract_file_path) + "is not a solidity file or file is not exist!")
                continue

            set_solc_version('v' + row['useversion'])

            f = open(contract_file_path)
            file_content = f.read()
            f.close()

    #             evm_opcodes_maps = {}
            evm_bytecodes_maps = {}

#             set_solc_version('v0.4.24')
#             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
            result = compile_source(file_content)
            begin_blocks = []
            for k,v in result.items():
    #             print(len(v['bin-runtime']))
    #             print(v['bin-runtime'])
    #             print(v)
                if len(v['bin-runtime']) == 0:
                    continue
                cfg = CFG(v['bin-runtime'])
    #                 print(len(cfg.basic_blocks))
                for i in range(len(cfg.basic_blocks)):
                    basic_block = cfg.basic_blocks[i]
                    if len(basic_block.all_incoming_basic_blocks) == 0:
                        begin_blocks.append(basic_block)
    #                     evm_opcodes_list = []
                    evm_bytecodes_list = []
                    link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                    #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                    #                                           basic_block.end.pc))
                    #     print('\t\tInstructions:')
                    for ins in basic_block.instructions:
        #                 evm_opcodes_list.append(ins.name) #opcode
                        # evm_bytecodes_list.append(ins.name) #bytecode
                        evm_bytecodes_list.append(ins.opcode) #bytecode
    #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                        # if ins.operand is not None: #values
                        #     evm_bytecodes_list.append(ins.operand)
                    evm_bytecodes_maps[link_block] = evm_bytecodes_list
                del cfg;
                gc.collect()

    #             file_bytecodes = []
    #             for contract_index in result:
    #                 file_bytecodes.append(result[contract_index]['opcodes'])
    #             file_bytecodes_str = ' '.join(file_opcodes)

            #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
#             print(len(list(evm_bytecodes_maps.keys())))
    #                 print(evm_bytecodes_maps.keys())
    #         all_evm_opcodes_lists = []
            all_evm_bytecodes_lists = []
    #         all_blocks_name = []
    #                 max_num = 0           
#             print(len(begin_blocks))
            for begin_block in begin_blocks:
                all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

#             print(all_evm_bytecodes_lists)
            bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
            # print(len(bytecodes_lists[0]))
            files_bytecodes[file] = bytecodes_lists[:10] #10
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

    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)

                begin_blocks = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
            #                 evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values list
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    del cfg;
                    gc.collect()


                #先用全局的,不用function内部的,如果考虑函数之间的调用需要按函数存储然后重叠组合
    #                     print(len(list(evm_bytecodes_maps.keys())))
    #                     print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []

    #                     max_num = 0
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

    #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                # print(len(bytecodes_lists[0]))
                files_bytecodes[file] = bytecodes_lists[:10]
                compile_true += 1
                del bytecodes_lists, evm_bytecodes_maps, begin_blocks, all_evm_bytecodes_lists;
                gc.collect()
            except Exception as e:
                # print(e)
                false_files.append(file)
                print(file + " compiled faild!")
        # print(index)
        if (index+1) % 300 == 0:
            print("current file index: {}, success: {}".format(index+1, compile_true))
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open("./contract_bytecodes_list10_event.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(false_files, cls=NpEncoder)
    fileObject = open("./contract_falsecompile_list_event.json", 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_for_solidity_file(file_path, solc_version, instance_len):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            evm_bytecodes_maps = {}
            # evm_opcodes_maps = {}
            result = compile_source(file_content)
            begin_blocks = []

            contract_names = []
            for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                if len(v['bin-runtime']) == 0:
                    continue
                contract_names.append(k[8:])
                cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                for i in range(len(cfg.basic_blocks)):
                    basic_block = cfg.basic_blocks[i]
                    if len(basic_block.all_incoming_basic_blocks) == 0:
                        begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                    evm_bytecodes_list = []
                    # evm_opcodes_list = []
                    link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                    #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                    #                                           basic_block.end.pc))
                    #     print('\t\tInstructions:')
                    for ins in basic_block.instructions:
                        # evm_opcodes_list.append(ins.name) #opcode
                        # evm_bytecodes_list.append(ins.name) #bytecode
                        evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                        # if ins.operand is not None: #values
                        #     evm_bytecodes_list.append(ins.operand)
                    evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    # evm_opcodes_maps[link_block] = evm_opcodes_list
                del cfg;
                gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

        #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
            all_evm_bytecodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
        #             print(len(begin_blocks))
            for begin_block in begin_blocks:
                all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
            bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
            # print(len(bytecodes_lists[0]))
            return bytecodes_lists[:instance_len], contract_names #10
        except Exception:
        #             print("except:")
            try:
                set_solc_version('v0.4.24')
                f = open(file_path)
                file_content = f.read()
                f.close()

                evm_bytecodes_maps = {}
                # evm_opcodes_maps = {}

                result = compile_source(file_content)

                begin_blocks = []

                contract_names = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    contract_names.append(k[8:])
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        # evm_opcodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
                            # evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            # if ins.operand is not None: #values list
                            #     evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        # evm_opcodes_maps[link_block] = evm_opcodes_list
                    del cfg;
                    gc.collect()

        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []

        #                     max_num = 0
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

        #             print(all_evm_bytecodes_lists)
                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)

                # print(len(bytecodes_lists[0]))
                return bytecodes_lists[:instance_len], contract_names
            except Exception as e:
                print("compiled faild!")
                return None
    return None

def make_bytecodes_opcodes_for_solidity_file(file_path, solc_version, instance_len):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            evm_bytecodes_maps = {}
            evm_opcodes_maps = {}
            result = compile_source(file_content)
            begin_blocks = []

            contract_names = []
            for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                if len(v['bin-runtime']) == 0:
                    continue
                contract_names.append(k[8:])
                cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                for i in range(len(cfg.basic_blocks)):
                    basic_block = cfg.basic_blocks[i]
                    if len(basic_block.all_incoming_basic_blocks) == 0:
                        begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                    evm_bytecodes_list = []
                    evm_opcodes_list = []
                    link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                    #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                    #                                           basic_block.end.pc))
                    #     print('\t\tInstructions:')
                    for ins in basic_block.instructions:
                        evm_opcodes_list.append(ins.name) #opcode
                        # evm_bytecodes_list.append(ins.name) #bytecode
                        evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                        if ins.operand is not None: #values
                            evm_opcodes_list.append(str(hex(ins.operand)))
                            # evm_bytecodes_list.append(ins.operand)
                    
                    # if link_block in evm_bytecodes_maps:
                    #     print(link_block)
                    #     print("---------***************------------")
                    evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    evm_opcodes_maps[link_block] = evm_opcodes_list
                del cfg
                gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

        #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
            
            all_evm_bytecodes_lists = []
            all_evm_opcodes_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
            # print(len(begin_blocks))
            for begin_block in begin_blocks:
                all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val = cycle_research_continue2_op_and_byte([], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, [], [])
                all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

        #             print(all_evm_bytecodes_lists)
            # print(len(all_evm_bytecodes_lists))
            bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
            bytecodes_lists = []
            opcodes_lists = []
            # print(bytecodes_lists_ids)
            
            for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])
            # print(len(bytecodes_lists[0]))
            return bytecodes_lists, contract_names, opcodes_lists #10
        except Exception:
        #             print("except:")
            try:
                set_solc_version('v0.4.24')
                f = open(file_path)
                file_content = f.read()
                f.close()

                evm_bytecodes_maps = {}
                evm_opcodes_maps = {}

                result = compile_source(file_content)

                begin_blocks = []

                contract_names = []
                for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    contract_names.append(k[8:])
                    cfg = CFG(v['bin-runtime'])
        #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)):
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append(basic_block)
        #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        evm_opcodes_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
                            evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            if ins.operand is not None: #values list
                                evm_opcodes_list.append(str(hex(ins.operand)))
                                # evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        evm_opcodes_maps[link_block] = evm_opcodes_list
                    del cfg;
                    gc.collect()

        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
                all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
        #         all_blocks_name = []

        #                     max_num = 0
                for begin_block in begin_blocks:
                    all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val = cycle_research_continue2_op_and_byte([], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, [], [])
                    all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                    all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

        #             print(all_evm_bytecodes_lists)
                bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
                bytecodes_lists = []
                opcodes_lists = []
                for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                    bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                    opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])
                # print(len(bytecodes_lists[0]))
                return bytecodes_lists, contract_names, opcodes_lists #10
            except Exception as e:
                print("compiled faild!")
                return None
    return None

def _get_positions(c_asm):  
    asm = c_asm['.data']['0'].copy()

    positions = asm['.code'].copy()
    while(True):
        try:
            positions.append(None)
            positions += asm['.data']['0']['.code'].copy()
            asm = asm['.data']['0'].copy()
        except:
            break
    return positions

def _get_positions_pcs_mapping(positions, instructions):
    map_pcs = {}
    
    inter_positions = 0
    inter_instructions = 0
    current_instructions = True
    
    position_name = None
    position_value = None
    
    instruction_name = None
    instruction_value = None
    
    while (inter_positions < len(positions) and (inter_instructions < len(instructions))) :
        if (not positions[inter_positions]) or positions[inter_positions]['name'].startswith("tag"):
            inter_positions += 1
            continue
        position_name = positions[inter_positions]['name']
        position_value = None
                
        if position_name in ["JUMP", "PUSH [tag]", "KECCAK256", "INVALID", "SELFDESTRUCT", "PUSHDEPLOYADDRESS"]:
            if position_name == "PUSH [tag]":
                position_name = position_name[:4]
                position_value = "*"
            elif position_name == "KECCAK256":
                position_name = "SHA3"
            elif position_name == "INVALID":
                position_name = "ASSERTFAIL"
            elif position_name == "SELFDESTRUCT":
                position_name = "SUICIDE"
            elif position_name == "PUSHDEPLOYADDRESS":
                position_name = position_name[:4]
                position_value = "*"
        elif 'value' in positions[inter_positions]:
            position_value = positions[inter_positions]['value']
        
        if current_instructions:
            instruction_name = instructions[inter_instructions].name
            instruction_value = None
            if instruction_name.startswith("PUSH"):
                instruction_name = instruction_name[:4]
            elif instruction_name in ["INVALID", "SELFDESTRUCT"]:
                if instruction_name == "INVALID":
                    instruction_name = "ASSERTFAIL"
                elif instruction_name == "SELFDESTRUCT":
                    instruction_name = "SUICIDE"
            if instructions[inter_instructions].has_operand:
                instruction_value = str(hex(instructions[inter_instructions].operand))[2:].upper()
            current_instructions = False
                        
        if (instruction_name == position_name) and ((instruction_value == position_value) or (position_value == "*")):
            map_pcs[instructions[inter_instructions].pc] = {'begin': positions[inter_positions]['begin'], 'end': positions[inter_positions]['end']}
#             map_pcs[inter_instructions] = inter_positions
            inter_instructions += 1
            inter_positions += 1

            current_instructions = True
        else:
            inter_positions += 1
                    
    if inter_instructions < len(instructions):
#         map_pcs[inter_instructions] = map_pcs[inter_instructions - 1]
        map_pcs[instructions[inter_instructions].pc] = map_pcs[instructions[inter_instructions - 1].pc]
        inter_instructions += 1
        while (inter_instructions < len(instructions)):
            map_pcs[instructions[inter_instructions].pc] = {'begin': -1, 'end': -1}
            inter_instructions += 1
            
#     print(map_pcs)
    return map_pcs

def make_bytecodes_opcodes_mapping_for_solidity_file(file_path, solc_version, instance_len):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            evm_bytecodes_maps = {}
            evm_opcodes_maps = {}
            evm_pc_maps = {}

            result = compile_source(file_content)
            begin_blocks = []

            contract_names = []
            pcs_mapping_contracts = {}

            for k,v in result.items():
        #             print(len(v['bin-runtime']))
        #             print(v['bin-runtime'])
        #             print(v)
                if len(v['bin-runtime']) == 0:
                    continue
                contract_names.append(k[8:])
                cfg = CFG(v['bin-runtime'])

                positions = _get_positions(v['asm'])

                map_pcs = None
                try:
                    map_pcs = _get_positions_pcs_mapping(positions, cfg.instructions)
                    # print("map_pcs", len(list(map_pcs.keys())))
                    if len(list(map_pcs.keys())) != len(cfg.instructions):
                        print("Warning!!! Length is not matching!!!")
                except:
                    print("Error!!! Length is not matching!!!")

                pcs_mapping_contracts[k] = map_pcs

        #                 print(len(cfg.basic_blocks))
                for i in range(len(cfg.basic_blocks)): 
                    basic_block = cfg.basic_blocks[i]
                    if len(basic_block.all_incoming_basic_blocks) == 0:
                        begin_blocks.append((k, basic_block))
        #                     evm_opcodes_list = []
                    evm_bytecodes_list = []
                    evm_opcodes_list = []
                    evm_pc_list = []
                    link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                    #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                    #                                           basic_block.end.pc))
                    #     print('\t\tInstructions:')
                    for ins in basic_block.instructions:
                        evm_pc_list.append(ins.pc)
                        evm_opcodes_list.append(ins.name) #opcode
                        # evm_bytecodes_list.append(ins.name) #bytecode
                        evm_bytecodes_list.append(ins.opcode) #bytecode
        #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                        if ins.operand is not None: #values
                            evm_opcodes_list.append(str(hex(ins.operand)))
                            # evm_bytecodes_list.append(ins.operand)
                    # if link_block in evm_bytecodes_maps:
                    #     if evm_bytecodes_list == evm_bytecodes_maps[link_block]:
                    #         print("--------------------------------", link_block, True)
                    #     else:
                    #         print("--------------------------------", link_block, False)
                    #         print(evm_bytecodes_list)
                    #         print(evm_bytecodes_maps[link_block])
                    evm_bytecodes_maps[link_block] = evm_bytecodes_list
                    evm_opcodes_maps[link_block] = evm_opcodes_list
                    evm_pc_maps[link_block] = evm_pc_list
                del cfg
                gc.collect()

        #             file_bytecodes = []
        #             for contract_index in result:
        #                 file_bytecodes.append(result[contract_index]['opcodes'])
        #             file_bytecodes_str = ' '.join(file_opcodes)

        #             print(len(list(evm_bytecodes_maps.keys())))
        #                 print(evm_bytecodes_maps.keys())
        #         all_evm_opcodes_lists = []
            all_evm_bytecodes_lists = []
            all_evm_opcodes_lists = []
            all_evm_pc_lists = []
            all_evm_contracts_lists = []
        #         all_blocks_name = []
        #                 max_num = 0           
        #             print(len(begin_blocks))
            for contract_index, begin_block in begin_blocks:
                all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = cycle_research_continue2_op_and_byte_and_pc([], [], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, evm_pc_maps, [], [], [])
                all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
                all_evm_pc_lists.extend(all_evm_pc_lists_val)
                all_evm_contracts_lists.extend([contract_index] * len(all_evm_opcodes_lists_val))

        #             print(all_evm_bytecodes_lists)
            bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
            bytecodes_lists = []
            opcodes_lists = []
            pcs_lists = []
            for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])

                pcs_mapping_val = pcs_mapping_contracts[all_evm_contracts_lists[bytecodes_lists_ids[ii]]]
                pcs_lists_val = []
                for pc_val in all_evm_pc_lists[bytecodes_lists_ids[ii]]:
                    pcs_lists_val.append(pcs_mapping_val[pc_val])

                # print(len(all_evm_pc_lists[bytecodes_lists_ids[ii]]))
                # print(len(pcs_lists_val))

                pcs_lists.append(pcs_lists_val)
            # print(len(bytecodes_lists[0]))
            return bytecodes_lists, contract_names, opcodes_lists, pcs_lists #10
        except Exception:
        #             print("except:")
            try:
                set_solc_version('v0.4.24')
                f = open(file_path)
                file_content = f.read()
                f.close()

                evm_bytecodes_maps = {}
                evm_opcodes_maps = {}
                evm_pc_maps = {}

                result = compile_source(file_content)

                begin_blocks = []

                contract_names = []
                pcs_mapping_contracts = {}
                for k,v in result.items():
            #             print(len(v['bin-runtime']))
            #             print(v['bin-runtime'])
            #             print(v)
                    if len(v['bin-runtime']) == 0:
                        continue
                    contract_names.append(k[8:])
                    cfg = CFG(v['bin-runtime'])

                    positions = _get_positions(v['asm'])

                    map_pcs = None
                    try:
                        map_pcs = _get_positions_pcs_mapping(positions, cfg.instructions)
                        print("map_pcs", len(list(map_pcs.keys())))
                        if len(list(map_pcs.keys())) != len(cfg.instructions):
                            print("Warning!!! Length is not matching!!!")
                    except:
                        print("Error!!! Length is not matching!!!")

                    pcs_mapping_contracts[k] = map_pcs

            #                 print(len(cfg.basic_blocks))
                    for i in range(len(cfg.basic_blocks)): 
                        basic_block = cfg.basic_blocks[i]
                        if len(basic_block.all_incoming_basic_blocks) == 0:
                            begin_blocks.append((k, basic_block))
            #                     evm_opcodes_list = []
                        evm_bytecodes_list = []
                        evm_opcodes_list = []
                        evm_pc_list = []
                        link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
                        #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
                        #                                           basic_block.end.pc))
                        #     print('\t\tInstructions:')
                        for ins in basic_block.instructions:
                            evm_pc_list.append(ins.pc)
                            evm_opcodes_list.append(ins.name) #opcode
                            # evm_bytecodes_list.append(ins.name) #bytecode
                            evm_bytecodes_list.append(ins.opcode) #bytecode
            #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                            if ins.operand is not None: #values
                                evm_opcodes_list.append(str(hex(ins.operand)))
                                # evm_bytecodes_list.append(ins.operand)
                        evm_bytecodes_maps[link_block] = evm_bytecodes_list
                        evm_opcodes_maps[link_block] = evm_opcodes_list
                        evm_pc_maps[link_block] = evm_pc_list
                    del cfg;
                    gc.collect()

        #                     print(len(list(evm_bytecodes_maps.keys())))
        #                     print(evm_bytecodes_maps.keys())
                all_evm_opcodes_lists = []
                all_evm_bytecodes_lists = []
                all_evm_pc_lists = []
                all_evm_contracts_lists = []
        #         all_blocks_name = []

        #                     max_num = 0
                for contract_index, begin_block in begin_blocks:
                    all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = cycle_research_continue2_op_and_byte_and_pc([], [], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, evm_pc_maps, [], [], [])
                    all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                    all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
                    all_evm_pc_lists.extend(all_evm_pc_lists_val)
                    all_evm_contracts_lists.extend([contract_index] * len(all_evm_opcodes_lists_val))

        #             print(all_evm_bytecodes_lists)
                bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
                bytecodes_lists = []
                opcodes_lists = []
                pcs_lists = []
                for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                    bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                    opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])

                    pcs_mapping_val = pcs_mapping_contracts[all_evm_contracts_lists[bytecodes_lists_ids[ii]]]
                    pcs_lists_val = []
                    for pc_val in all_evm_pc_lists[bytecodes_lists_ids[ii]]:
                        pcs_lists_val.append(pcs_mapping_val[pc_val])
                    
                    # print(len(all_evm_pc_lists[bytecodes_lists_ids[ii]]))
                    # print(len(pcs_lists_val))

                    pcs_lists.append(pcs_lists_val)
                # print(len(bytecodes_lists[0]))
                return bytecodes_lists, contract_names, opcodes_lists, pcs_lists #10
            except Exception as e:
                print("compiled faild!")
                return None
    return None

def make_bytecodes_for_binary_file(file_path):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()

        code = file_content.strip()
    
        evm_bytecodes_maps = {}
        begin_blocks = []
        
        cfg = CFG(code)
#                 print(len(cfg.basic_blocks))
        for ii in range(len(cfg.basic_blocks)):
            basic_block = cfg.basic_blocks[ii]
            if len(basic_block.all_incoming_basic_blocks) == 0:
                begin_blocks.append(basic_block)
#                     evm_opcodes_list = []
            evm_bytecodes_list = []
            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
            #                                           basic_block.end.pc))
            #     print('\t\tInstructions:')
            for ins in basic_block.instructions:
#                 evm_opcodes_list.append(ins.name) #opcode
                # evm_bytecodes_list.append(ins.name) #bytecode
                evm_bytecodes_list.append(ins.opcode) #bytecode
#                     evm_opcodes_maps[link_block] = evm_opcodes_list
                # if ins.operand is not None: #values
                #     evm_bytecodes_list.append(ins.operand)
            evm_bytecodes_maps[link_block] = evm_bytecodes_list

        all_evm_bytecodes_lists = []

        for begin_block in begin_blocks:
            all_evm_bytecodes_lists.extend(cycle_research_continue2([], [], [], begin_block, 0, evm_bytecodes_maps, []))

#             print(all_evm_bytecodes_lists)
        bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
        # print(len(bytecodes_lists[0]))
        return bytecodes_lists[:10] #10
    except Exception as e:
        print("construct failed!")
        return None

def make_bytecodes_opcodes_for_binary_file(file_path, instance_len):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()

        code = file_content.strip()
    
        evm_bytecodes_maps = {}
        evm_opcodes_maps ={}
        begin_blocks = []

        contract_names = ['BytecodeContract', '...']

        cfg = CFG(code)
    #                 print(len(cfg.basic_blocks))
        for i in range(len(cfg.basic_blocks)):
            basic_block = cfg.basic_blocks[i]
            if len(basic_block.all_incoming_basic_blocks) == 0:
                begin_blocks.append(basic_block)
    #                     evm_opcodes_list = []
            evm_bytecodes_list = []
            evm_opcodes_list = []
            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
            #                                           basic_block.end.pc))
            #     print('\t\tInstructions:')
            for ins in basic_block.instructions:
                evm_opcodes_list.append(ins.name) #opcode
                # evm_bytecodes_list.append(ins.name) #bytecode
                evm_bytecodes_list.append(ins.opcode) #bytecode
    #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                if ins.operand is not None: #values
                    evm_opcodes_list.append(str(hex(ins.operand)))
                    # evm_bytecodes_list.append(ins.operand)
            evm_bytecodes_maps[link_block] = evm_bytecodes_list
            evm_opcodes_maps[link_block] = evm_opcodes_list

        all_evm_opcodes_lists = []
        all_evm_bytecodes_lists = []
#         all_blocks_name = []
#                     max_num = 0
        for begin_block in begin_blocks:
            all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val = cycle_research_continue2_op_and_byte([], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, [], [])
            all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
            all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

#             print(all_evm_bytecodes_lists)
        bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
        bytecodes_lists = []
        opcodes_lists = []
        for ii in range(min(instance_len, len(bytecodes_lists_ids))):
            bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
            opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])
        # print(len(bytecodes_lists[0]))
        return bytecodes_lists, contract_names, opcodes_lists #10
    except Exception as e:
        print("construct failed!")
        return None

def translate_bytecodes_to_opcodes(opcodes):
    opcodes_val = opcodes.strip().split(' ')
    map_replace = {'KECCAK256': 'SHA3', 'PC': 'GETPC'}

    i = 0
    opcodes_val_length = len(opcodes_val)
    bytecode_val = "0x"
    while i <  opcodes_val_length:
        try:
            if opcodes_val[i] == '':
                i = i + 1
            else:
                if opcodes_val[i][:2] == '0x':
                    bytecode_val = bytecode_val + opcodes_val[i][2:]
                elif opcodes_val[i] in ['KECCAK256', 'PC']:
                    hex_val = assemble_hex(map_replace[opcodes_val[i]])
                    bytecode_val = bytecode_val + hex_val[2:]
                else:
                    hex_val = assemble_hex(opcodes_val[i])
                    bytecode_val = bytecode_val + hex_val[2:]
                i = i + 1
        except:
            hex_val = assemble_hex(opcodes_val[i] + ' ' + opcodes_val[i+1])
            bytecode_val = bytecode_val + hex_val[2:]
            i = i + 2

    return bytecode_val

def make_bytecodes_opcodes_for_opcode_file(file_path, instance_len):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()
        
        code = translate_bytecodes_to_opcodes(file_content)

        evm_bytecodes_maps = {}
        evm_opcodes_maps ={}
        begin_blocks = []

        contract_names = ['OpcodeContract', '...']

        cfg = CFG(code)
    #                 print(len(cfg.basic_blocks))
        for i in range(len(cfg.basic_blocks)):
            basic_block = cfg.basic_blocks[i]
            if len(basic_block.all_incoming_basic_blocks) == 0:
                begin_blocks.append(basic_block)
    #                     evm_opcodes_list = []
            evm_bytecodes_list = []
            evm_opcodes_list = []
            link_block = '@{:x}-{:x}'.format(basic_block.start.pc, basic_block.end.pc)
            #     print('\t- @{:x}-{:x}'.format(basic_block.start.pc,
            #                                           basic_block.end.pc))
            #     print('\t\tInstructions:')
            for ins in basic_block.instructions:
                evm_opcodes_list.append(ins.name) #opcode
                # evm_bytecodes_list.append(ins.name) #bytecode
                evm_bytecodes_list.append(ins.opcode) #bytecode
    #                     evm_opcodes_maps[link_block] = evm_opcodes_list
                if ins.operand is not None: #values
                    evm_opcodes_list.append(str(hex(ins.operand)))
                    # evm_bytecodes_list.append(ins.operand)
            evm_bytecodes_maps[link_block] = evm_bytecodes_list
            evm_opcodes_maps[link_block] = evm_opcodes_list

        all_evm_opcodes_lists = []
        all_evm_bytecodes_lists = []
#         all_blocks_name = []
#                     max_num = 0
        for begin_block in begin_blocks:
            all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val = cycle_research_continue2_op_and_byte([], [], [], begin_block, 0, evm_bytecodes_maps, evm_opcodes_maps, [], [])
            all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
            all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

#             print(all_evm_bytecodes_lists)
        bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
        bytecodes_lists = []
        opcodes_lists = []
        for ii in range(min(instance_len, len(bytecodes_lists_ids))):
            bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
            opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])
        # print(len(bytecodes_lists[0]))
        return bytecodes_lists, contract_names, opcodes_lists #10
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
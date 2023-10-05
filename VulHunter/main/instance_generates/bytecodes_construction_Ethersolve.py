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

import pyevmasm
import subprocess as sp

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

class CFGBlock:
    def __init__(self, evm_bytecodes_list, evm_opcodes_list, evm_pc_list):
        self.all_outgoing_basic_blocks = []
        self.all_incoming_basic_blocks = []

        self.evm_bytecodes_list = evm_bytecodes_list
        self.evm_opcodes_list = evm_opcodes_list
        self.evm_pc_list = evm_pc_list
    
    def string_map(self):
        return {'all_outgoing_basic_blocks': self.all_outgoing_basic_blocks, 'all_incoming_basic_blocks': self.all_incoming_basic_blocks, 'evm_bytecodes_list': self.evm_bytecodes_list, 'evm_pc_list': self.evm_pc_list}

class ContractInstance:
    CFGBuilder_path = None
    runtimeBytecode_path = None
    CFG_path = None
    CFG_dot_path = None
    block_edges = None
    # pcs = None

    def __init__(self, CFGBuilder_path : str, output_dir='./output'):
        self.CFGBuilder_path = CFGBuilder_path
        self.output_dir = output_dir

    def set_translate_path(self, runtimeBytecode_path : str, CFG_path = None) -> None:
        self.runtimeBytecode_path = runtimeBytecode_path
        if CFG_path == None:
            dir_name = os.path.splitext(runtimeBytecode_path.split('/')[-1])[0]
            self.CFG_path = os.path.join(self.output_dir, dir_name)
            self.CFG_dot_path = os.path.join(self.CFG_path, 'CFG.dot')
            self.CFG_path = os.path.join(self.CFG_path, 'CFG.json')
            print(self.CFG_path)
        else:
            self.CFG_path = CFG_path
            self.CFG_dot_path = os.path.splitext(CFG_path)[0] + '.dot'
    
    def buildCFG_deploy(self) -> None:
        if self.CFGBuilder_path != None and self.runtimeBytecode_path != None:
            build_command = "java -jar "+self.CFGBuilder_path+"/EtherSolve.jar " + self.runtimeBytecode_path + " -c -o" + self.CFG_path + " -j"
            sp.call(build_command, shell=True)
        else:
            print("[Failed]\nYou have to give both the path of the EtherSolve tool (flag: -B or --CFG-builder) and the path of a contract runtime bytecode (flag: -R or --bin_runtime)")
            sys.exit(0)
    
    def buildCFG_dot_deploy(self) -> None:
        if self.CFGBuilder_path != None and self.runtimeBytecode_path != None:
            build_command = "java -jar "+self.CFGBuilder_path+"/EtherSolve.jar " + self.runtimeBytecode_path + " -c -o" + self.CFG_dot_path + " -d"
            sp.call(build_command, shell=True)

            CFG_png_path = os.path.join('/'.join(self.CFG_dot_path.split('/')[:-1]), 'CFG.png')
            visul_command = 'dot -T png {} -o {}'.format(self.CFG_dot_path, CFG_png_path)
            sp.call(visul_command, shell=True)
        else:
            print("[Failed]\nYou have to give both the path of the EtherSolve tool (flag: -B or --CFG-builder) and the path of a contract runtime bytecode (flag: -R or --bin_runtime)")
            sys.exit(0)
    
    def buildCFG_runtime(self) -> None:
        if self.CFGBuilder_path != None and self.runtimeBytecode_path != None:
            build_command = "java -jar "+self.CFGBuilder_path+"/EtherSolve.jar " + self.runtimeBytecode_path + " -r -o" + self.CFG_path + " -j"
            sp.call(build_command, shell=True)
        else:
            print("[Failed]\nYou have to give both the path of the EtherSolve tool (flag: -B or --CFG-builder) and the path of a contract runtime bytecode (flag: -R or --bin_runtime)")
            sys.exit(0)
    
    def buildCFG_dot_runtime(self) -> None:
        if self.CFGBuilder_path != None and self.runtimeBytecode_path != None:
            build_command = "java -jar "+self.CFGBuilder_path+"/EtherSolve.jar " + self.runtimeBytecode_path + " -r -o" + self.CFG_dot_path + " -d"
            # print(build_command)
            sp.call(build_command, shell=True)

            CFG_png_path = os.path.join('/'.join(self.CFG_dot_path.split('/')[:-1]), 'CFG.png')
            visul_command = 'dot -T png {} -o {}'.format(self.CFG_dot_path, CFG_png_path)
            # print(visul_command)
            sp.call(visul_command, shell=True)
        else:
            print("[Failed]\nYou have to give both the path of the EtherSolve tool (flag: -B or --CFG-builder) and the path of a contract runtime bytecode (flag: -R or --bin_runtime)")
            sys.exit(0)
    
    def parseCFG(self, cfg_path = None) -> None:
        
        cfg_json = None

        if cfg_path == None:
            with open(self.CFG_path) as cr:
                cfg_json = json.load(cr)
        else:
            with open(cfg_path) as cr:
                cfg_json = json.load(cr)

        # evm_bytecodes_maps = {}
        # evm_opcodes_maps = {}
        # evm_pc_maps = {}

        block_edges = {}
        
        if 'runtimeCfg' in cfg_json:
            block_pc_names = {}
            # pcs = []

            nodes = cfg_json['runtimeCfg']['nodes']

            exit_block_offset = -1

            for node in nodes:
                binaryHash = cfg_json['binaryHash']
                link_block = '{}@{:x}-{:x}'.format(binaryHash, node['offset'], node['offset']+node['length']-1)
                
                block_pc_names[node['offset']] = link_block 

                if node['type'] == 'exit':
                    exit_block_offset = node['offset']
                    continue

                evm_bytecodes_list = []
                # evm_opcodes_list = []
                # evm_pc_list = []

                for insts in node['parsedOpcodes'].split('\n'):
                    inter_index = insts.find(':')
                    # pc = int(insts[:inter_index])
                    insts_val = insts[inter_index+2:]
                    insts_vals = insts_val.split(' ')

                    # evm_pc_list.append(pc)

                    if len(insts_vals) > 1:
                        evm_bytecodes_list.append(int(assemble_hex(insts_val)[:4], 16))

                        # evm_opcodes_list.append(insts_vals[0])
                        # evm_opcodes_list.append(insts_vals[1])

                        # value = insts_vals[1][2:].lstrip('0')
                        # if value == '':
                        #     value = '0'

                        # pcs.append((pc, insts_vals[0], value))
                    else:
                        evm_bytecodes_list.append(int(assemble_hex(insts_vals[0]), 16))

                        # evm_opcodes_list.append(insts_vals[0])

                        # pcs.append((pc, insts_vals[0]))
                
                # evm_bytecodes_maps[link_block] = evm_bytecodes_list
                # evm_opcodes_maps[link_block] = evm_opcodes_list
                # evm_pc_maps[link_block] = evm_pc_list

                block_edges[link_block] = CFGBlock(evm_bytecodes_list, None, None)

            # self.pcs = sorted(pcs, key=lambda k: k[0])

            edges = cfg_json['runtimeCfg']['successors']
            for edge in edges:
                if edge['from'] == exit_block_offset:
                    continue
                from_node_names = block_pc_names[edge['from']]
                for to_node in set(edge['to']):
                    if to_node == exit_block_offset:
                        continue
                    to_node_names = block_pc_names[to_node]
                    block_edges[from_node_names].all_outgoing_basic_blocks.append(to_node_names)
                    # if from_node_names in block_edges[to_node_names].all_incoming_basic_blocks:
                    #     print("error!", edge['from'], to_node)
                    block_edges[to_node_names].all_incoming_basic_blocks.append(from_node_names)
        else:
            print('[Warning]\nNone nodes in the bytecode file!')
                        
        self.block_edges = block_edges

    def parseCFG_op(self, cfg_path = None) -> None:
        
        cfg_json = None

        if cfg_path == None:
            with open(self.CFG_path) as cr:
                cfg_json = json.load(cr)
        else:
            with open(cfg_path) as cr:
                cfg_json = json.load(cr)

        # evm_bytecodes_maps = {}
        # evm_opcodes_maps = {}
        # evm_pc_maps = {}

        block_edges = {}
        
        if 'runtimeCfg' in cfg_json:
            block_pc_names = {}
            # pcs = []

            nodes = cfg_json['runtimeCfg']['nodes']

            exit_block_offset = -1

            for node in nodes:
                binaryHash = cfg_json['binaryHash']
                link_block = '{}@{:x}-{:x}'.format(binaryHash, node['offset'], node['offset']+node['length']-1)
                
                block_pc_names[node['offset']] = link_block 

                if node['type'] == 'exit':
                    exit_block_offset = node['offset']
                    continue

                evm_bytecodes_list = []
                evm_opcodes_list = []
                evm_pc_list = []

                for insts in node['parsedOpcodes'].split('\n'):
                    inter_index = insts.find(':')
                    pc = int(insts[:inter_index])
                    insts_val = insts[inter_index+2:]
                    insts_vals = insts_val.split(' ')

                    evm_pc_list.append(pc)

                    if len(insts_vals) > 1:
                        evm_bytecodes_list.append(int(assemble_hex(insts_val)[:4], 16))

                        evm_opcodes_list.append(insts_vals[0])
                        evm_opcodes_list.append(insts_vals[1])

                        # value = insts_vals[1][2:].lstrip('0')
                        # if value == '':
                        #     value = '0'

                        # pcs.append((pc, insts_vals[0], value))
                    else:
                        evm_bytecodes_list.append(int(assemble_hex(insts_vals[0]), 16))

                        evm_opcodes_list.append(insts_vals[0])

                        # pcs.append((pc, insts_vals[0]))
                
                # evm_bytecodes_maps[link_block] = evm_bytecodes_list
                # evm_opcodes_maps[link_block] = evm_opcodes_list
                # evm_pc_maps[link_block] = evm_pc_list

                block_edges[link_block] = CFGBlock(evm_bytecodes_list, evm_opcodes_list, evm_pc_list)

            # self.pcs = sorted(pcs, key=lambda k: k[0])

            edges = cfg_json['runtimeCfg']['successors']
            for edge in edges:
                if edge['from'] == exit_block_offset:
                    continue
                from_node_names = block_pc_names[edge['from']]
                for to_node in set(edge['to']):
                    if to_node == exit_block_offset:
                        continue
                    to_node_names = block_pc_names[to_node]
                    block_edges[from_node_names].all_outgoing_basic_blocks.append(to_node_names)
                    # if from_node_names in block_edges[to_node_names].all_incoming_basic_blocks:
                    #     print("error!", edge['from'], to_node)
                    block_edges[to_node_names].all_incoming_basic_blocks.append(from_node_names)
        else:
            print('[Warning]\nNone nodes in the bytecode file!')
                        
        self.block_edges = block_edges
    
    max_val = 0

    def cycle_research_continue2_byte_and_pc(self, current_bytecodes_list, current_blocks_list, current_block_name, cycleblock_num, all_evm_bytecodes_lists):
        
        if current_block_name in current_blocks_list:
            cycleblock_num += 1
        else:
            cycleblock_num = 0

        # print(current_block_name)
        current_block = self.block_edges[current_block_name]

        current_blocks_list.append(current_block_name)
        current_bytecodes_list.extend(current_block.evm_bytecodes_list)
        # current_pc_list.extend(current_block.evm_pc_list)

        if cycleblock_num == 2 or len(current_blocks_list) >= 32:
            # all_evm_opcodes_lists.append(current_opcodes_list)
            all_evm_bytecodes_lists.append(current_bytecodes_list)
            # all_evm_pc_lists.append(current_pc_list)
    #         all_blocks_name.append(current_blocks_list)
        else:
            if len(current_block.all_outgoing_basic_blocks) == 0:
                # all_evm_opcodes_lists.append(current_opcodes_list)
                all_evm_bytecodes_lists.append(current_bytecodes_list)
                # all_evm_pc_lists.append(current_pc_list)
    #             all_blocks_name.append(current_blocks_list)
            else:
                for i in range(len(current_block.all_outgoing_basic_blocks)):
                    # if self.max_val < len(all_evm_bytecodes_lists):
                    #     self.max_val = len(all_evm_bytecodes_lists)
                    #     print(self.max_val)
                    if len(all_evm_bytecodes_lists) > 100000:
                        return all_evm_bytecodes_lists
                    else:
                    # all_evm_bytecodes_lists = self.cycle_research_continue2_op_and_byte_and_pc(current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, all_evm_bytecodes_lists)
                        all_evm_bytecodes_lists.extend(self.cycle_research_continue2_byte_and_pc(current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, []))

        return all_evm_bytecodes_lists

    def cycle_research_continue2_op_and_byte_and_pc(self, current_opcodes_list, current_bytecodes_list, current_pc_list, current_blocks_list, current_block_name, cycleblock_num, all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists):
        
        if current_block_name in current_blocks_list:
            cycleblock_num += 1
        else:
            cycleblock_num = 0

        # print(current_block_name)
        current_block = self.block_edges[current_block_name]

        current_blocks_list.append(current_block_name)
        current_bytecodes_list.extend(current_block.evm_bytecodes_list)
        current_opcodes_list.extend(current_block.evm_opcodes_list)
        current_pc_list.extend(current_block.evm_pc_list)

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
                    # if self.max_val < len(all_evm_bytecodes_lists):
                    #     self.max_val = len(all_evm_bytecodes_lists)
                    #     print(self.max_val)
                    if len(all_evm_bytecodes_lists) > 100000:
                        return all_evm_bytecodes_lists
                    else:
                    # all_evm_bytecodes_lists = self.cycle_research_continue2_op_and_byte_and_pc(current_bytecodes_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, all_evm_bytecodes_lists)
                        all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = self.cycle_research_continue2_op_and_byte_and_pc(current_opcodes_list.copy(), current_bytecodes_list.copy(), current_pc_list.copy(), current_blocks_list.copy(), current_block.all_outgoing_basic_blocks[i], cycleblock_num, [], [], [])
                        all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                        all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
                        all_evm_pc_lists.extend(all_evm_pc_lists_val)

        return all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists

    def traverse_cfg(self) -> list:
        self.parseCFG()

        # print("block", len(self.block_edges.keys()))

        # cc = {}
        # for block_edge_k,block_edge_v in self.block_edges.items():
        #     cc[block_edge_k] = {'all_outgoing_basic_blocks': block_edge_v.all_outgoing_basic_blocks, 'all_incoming_basic_blocks': block_edge_v.all_incoming_basic_blocks}

        # data_json = json.dumps(cc, cls=NpEncoder)
        # fileObject = open(os.path.splitext(self.CFG_path)[0] + '_blocks.json', 'w') #contract_bytecodes
        # fileObject.write(data_json)
        # fileObject.close()

        begin_blocks = []
        for block_edge_k,block_edge_v in self.block_edges.items():
            # print(block_edge_k, block_edge_v.string_map())
            if len(block_edge_v.all_incoming_basic_blocks) == 0:
                begin_blocks.append(block_edge_k)
        
        all_evm_bytecodes_lists = []
        # all_evm_opcodes_lists = []
        # all_evm_pc_lists = []
        # all_evm_contracts_lists = []

        for block_name in begin_blocks:
            all_evm_bytecodes_lists_val = self.cycle_research_continue2_byte_and_pc([], [], block_name, 0, [])
            all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
            # all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
            # all_evm_pc_lists.extend(all_evm_pc_lists_val)
            # all_evm_contracts_lists.extend([contract_index] * len(all_evm_opcodes_lists_val))
        
        return all_evm_bytecodes_lists
    
    def traverse_cfg_op(self) -> list:
        self.parseCFG_op()

        # print("block", len(self.block_edges.keys()))

        # cc = {}
        # for block_edge_k,block_edge_v in self.block_edges.items():
        #     cc[block_edge_k] = {'all_outgoing_basic_blocks': block_edge_v.all_outgoing_basic_blocks, 'all_incoming_basic_blocks': block_edge_v.all_incoming_basic_blocks}

        # data_json = json.dumps(cc, cls=NpEncoder)
        # fileObject = open(os.path.splitext(self.CFG_path)[0] + '_blocks.json', 'w') #contract_bytecodes
        # fileObject.write(data_json)
        # fileObject.close()

        begin_blocks = []
        for block_edge_k,block_edge_v in self.block_edges.items():
            # print(block_edge_k, block_edge_v.string_map())
            if len(block_edge_v.all_incoming_basic_blocks) == 0:
                begin_blocks.append(block_edge_k)
        
        all_evm_bytecodes_lists = []
        all_evm_opcodes_lists = []
        all_evm_pc_lists = []
        # all_evm_contracts_lists = []

        for block_name in begin_blocks:
            all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = self.cycle_research_continue2_op_and_byte_and_pc([], [], [], [], block_name, 0, [], [], [])
            all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
            all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
            all_evm_pc_lists.extend(all_evm_pc_lists_val)
            # all_evm_contracts_lists.extend([contract_index] * len(all_evm_opcodes_lists_val))
        
        return all_evm_bytecodes_lists, all_evm_opcodes_lists, all_evm_pc_lists

# import pdb

def make_bytecodes_continue2(contract_root_dir, solcversions_path, instance_len, tmp_dir, instance_dir, CFGBuilder_path):
    file_solc_versions = None
    with open(solcversions_path) as fp:
        file_solc_versions = json.load(fp)

    output_dir = tmp_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                    print(str(contract_file_path) + "is not .sol file!")
                    continue
                file_solc = "0.4.24"
                if (contract_file_name in file_solc_versions) and (file_solc_versions[contract_file_name] != None):
                    file_solc = file_solc_versions[contract_file_name]
                set_solc_version('v' + file_solc)

                f = open(contract_file_path)
                file_content = f.read()
                f.close()

        #             evm_opcodes_maps = {}
                all_evm_bytecodes_lists = []

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)

                output_contract_dir = os.path.join(output_dir, os.path.splitext(contract_file_name)[0])
                if not os.path.exists(output_contract_dir):
                #     continue
                # else:
                    os.makedirs(output_contract_dir)

                # pdb.set_trace() 

                for k,v in result.items():
                    # print(k)
                    # print('contract', k)
                    # print("Runtime-bin bytecode:")
                    # print(v['bin-runtime'])

                    binruntime = v['bin-runtime']

                    if len(binruntime) == 0:
                        continue

                    if '_' in binruntime:
                        print(contract_file_name, k, "exist the third library!")
                        # print(binruntime)
                        pat = re.compile(r'__(<stdin>.*?)__')
                        # pat = re.compile(r'__(.*[^_]?)_')
                        res = list(set(pat.findall(binruntime)))
                        print(res)
                        replace_maps = {}
                        for address_val in range(len(res)):
                            replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                        binruntime = link_code(binruntime, replace_maps)
                        # print(binruntime)

                    contract_name = k[8:]
                    file_dir = os.path.join(output_contract_dir, contract_name)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)

                    compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                    with open(compile_file_path, 'w') as f:
                        f.write(binruntime)

                    cfg_file_path = os.path.join(file_dir, 'CFG.json')
                    
                    contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                    contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                    # contractInstance.buildCFG_dot_runtime()
                    contractInstance.buildCFG_runtime()

                    all_evm_bytecodes_lists_val = contractInstance.traverse_cfg()
                    all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)

                    # print(contract_name, len(all_evm_pc_lists_val))

                    del contractInstance, all_evm_bytecodes_lists_val
                    gc.collect()

                bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                
                # data_json = json.dumps(bytecodes_lists, cls=NpEncoder)
                # fileObject = open(os.path.join(output_contract_dir, "instances.json"), 'w') #contract_bytecodes
                # fileObject.write(data_json)
                # fileObject.close()
                
                # print(len(bytecodes_lists[0]))
                files_bytecodes[contract_file_name] = bytecodes_lists[:instance_len] #10

                # print(contract_file_name, len(bytecodes_lists))
                compile_true += 1
                del bytecodes_lists, all_evm_bytecodes_lists;
                gc.collect()
            except Exception:
    #             print("except:")
                try:
                    set_solc_version('v0.4.24')
                    f = open(contract_file_path)
                    file_content = f.read()
                    f.close()

                    all_evm_bytecodes_lists = []
                    result = compile_source(file_content)

                    output_contract_dir = os.path.join(output_dir, os.path.splitext(contract_file_name)[0])
                    if not os.path.exists(output_contract_dir):
                        os.makedirs(output_contract_dir)

                    for k,v in result.items():
                        # print('contract', k)
                        # print("Runtime-bin bytecode:")
                        # print(v['bin-runtime'])

                        binruntime = v['bin-runtime']

                        if len(binruntime) == 0:
                            continue

                        if '_' in binruntime:
                            print(contract_file_name, k, "exist the third library!")
                            # print(binruntime)
                            pat = re.compile(r'__(<stdin>.*?)__')
                            # pat = re.compile(r'__(.*[^_]?)_')
                            res = list(set(pat.findall(binruntime)))
                            print(res)
                            replace_maps = {}
                            for address_val in range(len(res)):
                                replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                            binruntime = link_code(binruntime, replace_maps)
                            # print(binruntime)

                        contract_name = k[8:]
                        file_dir = os.path.join(output_contract_dir, contract_name)
                        if not os.path.exists(file_dir):
                            os.makedirs(file_dir)

                        compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                        with open(compile_file_path, 'w') as f:
                            f.write(binruntime)
                        
                        cfg_file_path = os.path.join(file_dir, 'CFG.json')

                        contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                        contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                        # contractInstance.buildCFG_dot_runtime()
                        contractInstance.buildCFG_runtime()

                        all_evm_bytecodes_lists_val = contractInstance.traverse_cfg()
                        all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)

                        # print(contract_name, len(all_evm_pc_lists_val))

                        del contractInstance, all_evm_bytecodes_lists_val
                        gc.collect()

                    bytecodes_lists = sorted(all_evm_bytecodes_lists, key=lambda d:len(d), reverse = True)
                    # print(len(bytecodes_lists[0]))
                    # data_json = json.dumps(bytecodes_lists, cls=NpEncoder)
                    # fileObject = open(os.path.join(output_contract_dir, "instances.json"), 'w') #contract_bytecodes
                    # fileObject.write(data_json)
                    # fileObject.close()
                    
                    # print(len(bytecodes_lists[0]))
                    files_bytecodes[contract_file_name] = bytecodes_lists[:instance_len] #10

                    # print(contract_file_name, len(bytecodes_lists))
                    compile_true += 1
                    del bytecodes_lists, all_evm_bytecodes_lists, data_json, fileObject;

                    gc.collect()
                except Exception as e:
                    print(e)
                    false_files.append(contract_file_name)
                    print(contract_file_name + " compiled faild!")
            index = index + 1
            # print(index)
            if index % 200 == 0:
                print("current file index: {}, success: {}".format(index, compile_true))
        #     if index == 5:
        #         break
        # break
        
    data_json = json.dumps(files_bytecodes, cls=NpEncoder)
    fileObject = open(os.path.join(instance_dir, "contract_bytecodes_list10_ethersolve.json"), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

    data_json = json.dumps(false_files, cls=NpEncoder)
    fileObject = open(os.path.join(instance_dir, "contract_falsecompile_list_ethersolve.json"), 'w') #contract_bytecodes
    fileObject.write(data_json)
    fileObject.close()

def make_bytecodes_opcodes_for_solidity_file(file_path, solc_version, instance_len, tmp_dir, CFGBuilder_path):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            contract_file_name = os.path.splitext(file_path.split('/')[-1])[0]

            all_evm_bytecodes_lists = []
            all_evm_opcodes_lists = []

            contract_names = []

    #             set_solc_version('v0.4.24')
    #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
            result = compile_source(file_content)

            output_contract_dir = os.path.join(tmp_dir, contract_file_name)
            if not os.path.exists(output_contract_dir):
            #     continue
            # else:
                os.makedirs(output_contract_dir)

            # pdb.set_trace() 

            for k,v in result.items():
                # print(k)
                # print('contract', k)
                # print("Runtime-bin bytecode:")
                # print(v['bin-runtime'])

                binruntime = v['bin-runtime']

                if len(binruntime) == 0:
                    continue

                if '_' in binruntime:
                    print(contract_file_name, k, "exist the third library!")
                    # print(binruntime)
                    pat = re.compile(r'__(<stdin>.*?)__')
                    # pat = re.compile(r'__(.*[^_]?)_')
                    res = list(set(pat.findall(binruntime)))
                    print(res)
                    replace_maps = {}
                    for address_val in range(len(res)):
                        replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                    binruntime = link_code(binruntime, replace_maps)
                    # print(binruntime)

                contract_name = k[8:]
                file_dir = os.path.join(output_contract_dir, contract_name)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                with open(compile_file_path, 'w') as f:
                    f.write(binruntime)

                cfg_file_path = os.path.join(file_dir, 'CFG.json')
                
                contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                # contractInstance.buildCFG_dot_runtime()
                contractInstance.buildCFG_runtime()

                all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
                all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

                # print(contract_name, len(all_evm_pc_lists_val))

                del contractInstance, all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val
                gc.collect()

            bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
            bytecodes_lists = []
            opcodes_lists = []

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

                contract_file_name = os.path.splitext(file_path.split('/')[-1])[0]

                all_evm_bytecodes_lists = []
                all_evm_opcodes_lists = []

                contract_names = []

        #             set_solc_version('v0.4.24')
        #             file_content = "pragma solidity 0.4.24; contract Foo { mapping (address => uint) balances; constructor() public {} function aa (uint cc) public returns(uint dd) {return cc;} function kk (uint cc) public returns(uint dd) {if(cc==1){return 1;}else{return kk(cc)+1;}}}"
                result = compile_source(file_content)

                output_contract_dir = os.path.join(tmp_dir, contract_file_name)
                if not os.path.exists(output_contract_dir):
                #     continue
                # else:
                    os.makedirs(output_contract_dir)

                # pdb.set_trace() 

                for k,v in result.items():
                    # print(k)
                    # print('contract', k)
                    # print("Runtime-bin bytecode:")
                    # print(v['bin-runtime'])

                    binruntime = v['bin-runtime']

                    if len(binruntime) == 0:
                        continue

                    if '_' in binruntime:
                        print(contract_file_name, k, "exist the third library!")
                        # print(binruntime)
                        pat = re.compile(r'__(<stdin>.*?)__')
                        # pat = re.compile(r'__(.*[^_]?)_')
                        res = list(set(pat.findall(binruntime)))
                        print(res)
                        replace_maps = {}
                        for address_val in range(len(res)):
                            replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                        binruntime = link_code(binruntime, replace_maps)
                        # print(binruntime)

                    contract_name = k[8:]
                    file_dir = os.path.join(output_contract_dir, contract_name)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)

                    compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                    with open(compile_file_path, 'w') as f:
                        f.write(binruntime)

                    cfg_file_path = os.path.join(file_dir, 'CFG.json')
                    
                    contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                    contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                    # contractInstance.buildCFG_dot_runtime()
                    contractInstance.buildCFG_runtime()

                    all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
                    all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                    all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)

                    # print(contract_name, len(all_evm_pc_lists_val))

                    del contractInstance, all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val
                    gc.collect()

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

def make_bytecodes_opcodes_mapping_for_solidity_file(file_path, solc_version, instance_len, tmp_dir, CFGBuilder_path):
    if os.path.splitext(file_path)[1] == '.sol':
        try:
            set_solc_version('v' + solc_version)

            f = open(file_path)
            file_content = f.read()
            f.close()

            contract_file_name = os.path.splitext(file_path.split('/')[-1])[0]

            all_evm_bytecodes_lists = []
            all_evm_opcodes_lists = []
            all_evm_pc_lists = []
            all_evm_contracts_lists = []

            contract_names = []
            pcs_mapping_contracts = {}

            result = compile_source(file_content)

            output_contract_dir = os.path.join(tmp_dir, contract_file_name)
            if not os.path.exists(output_contract_dir):
            #     continue
            # else:
                os.makedirs(output_contract_dir)

            # pdb.set_trace() 

            for k,v in result.items():
                # print(k)
                # print('contract', k)
                # print("Runtime-bin bytecode:")
                # print(v['bin-runtime'])

                binruntime = v['bin-runtime']

                if len(binruntime) == 0:
                    continue

                if '_' in binruntime:
                    print(contract_file_name, k, "exist the third library!")
                    # print(binruntime)
                    pat = re.compile(r'__(<stdin>.*?)__')
                    # pat = re.compile(r'__(.*[^_]?)_')
                    res = list(set(pat.findall(binruntime)))
                    print(res)
                    replace_maps = {}
                    for address_val in range(len(res)):
                        replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                    binruntime = link_code(binruntime, replace_maps)
                    # print(binruntime)

                contract_name = k[8:]
                file_dir = os.path.join(output_contract_dir, contract_name)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                with open(compile_file_path, 'w') as f:
                    f.write(binruntime)

                cfg_file_path = os.path.join(file_dir, 'CFG.json')
                
                contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                # contractInstance.buildCFG_dot_runtime()
                contractInstance.buildCFG_runtime()

                all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
                all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
                all_evm_pc_lists.extend(all_evm_pc_lists_val)
                all_evm_contracts_lists.extend([contract_name] * len(all_evm_opcodes_lists_val))
                # print(contract_name, len(all_evm_pc_lists_val))

                pcs_mapping_contracts[contract_name] = _get_positions_map(v['asm'])


                del contractInstance, all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val
                gc.collect()

            bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
            bytecodes_lists = []
            opcodes_lists = []
            pc_lists = []

            for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])

                pcs_mapping_val = pcs_mapping_contracts[all_evm_contracts_lists[bytecodes_lists_ids[ii]]]
                pc_lists_val = []
                for pc_val in all_evm_pc_lists[bytecodes_lists_ids[ii]]:
                    if pc_val in pcs_mapping_val:
                        pc_lists_val.append(pcs_mapping_val[pc_val])
                    else:
                        pc_lists_val.append({'begin': -1, 'end': -1})

                pc_lists.append(pc_lists_val)

            # print(len(bytecodes_lists[0]))
            return bytecodes_lists, contract_names, opcodes_lists, pc_lists #10
        except Exception:
        #             print("except:")
            try:
                set_solc_version('v0.4.24')

                f = open(file_path)
                file_content = f.read()
                f.close()

                contract_file_name = os.path.splitext(file_path.split('/')[-1])[0]

                all_evm_bytecodes_lists = []
                all_evm_opcodes_lists = []
                all_evm_pc_lists = []
                all_evm_contracts_lists = []

                contract_names = []
                pcs_mapping_contracts = {}

                result = compile_source(file_content)

                output_contract_dir = os.path.join(tmp_dir, contract_file_name)
                if not os.path.exists(output_contract_dir):
                #     continue
                # else:
                    os.makedirs(output_contract_dir)

                # pdb.set_trace() 

                for k,v in result.items():
                    # print(k)
                    # print('contract', k)
                    # print("Runtime-bin bytecode:")
                    # print(v['bin-runtime'])

                    binruntime = v['bin-runtime']

                    if len(binruntime) == 0:
                        continue

                    if '_' in binruntime:
                        print(contract_file_name, k, "exist the third library!")
                        # print(binruntime)
                        pat = re.compile(r'__(<stdin>.*?)__')
                        # pat = re.compile(r'__(.*[^_]?)_')
                        res = list(set(pat.findall(binruntime)))
                        print(res)
                        replace_maps = {}
                        for address_val in range(len(res)):
                            replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
                        binruntime = link_code(binruntime, replace_maps)
                        # print(binruntime)

                    contract_name = k[8:]
                    file_dir = os.path.join(output_contract_dir, contract_name)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)

                    compile_file_path = os.path.join(file_dir, contract_name + '.bin')
                    with open(compile_file_path, 'w') as f:
                        f.write(binruntime)

                    cfg_file_path = os.path.join(file_dir, 'CFG.json')
                    
                    contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
                    contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
                    # contractInstance.buildCFG_dot_runtime()
                    contractInstance.buildCFG_runtime()

                    all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
                    all_evm_bytecodes_lists.extend(all_evm_bytecodes_lists_val)
                    all_evm_opcodes_lists.extend(all_evm_opcodes_lists_val)
                    all_evm_pc_lists.extend(all_evm_pc_lists_val)
                    all_evm_contracts_lists.extend([contract_name] * len(all_evm_opcodes_lists_val))
                    # print(contract_name, len(all_evm_pc_lists_val))

                    pcs_mapping_contracts[contract_name] = _get_positions_map(v['asm'])


                    del contractInstance, all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val
                    gc.collect()

                bytecodes_lists_ids = sorted(range(len(all_evm_bytecodes_lists)), key=lambda x:len(all_evm_bytecodes_lists[x]), reverse = True)
                bytecodes_lists = []
                opcodes_lists = []
                pc_lists = []

                for ii in range(min(instance_len, len(bytecodes_lists_ids))):
                    bytecodes_lists.append(all_evm_bytecodes_lists[bytecodes_lists_ids[ii]])
                    opcodes_lists.append(all_evm_opcodes_lists[bytecodes_lists_ids[ii]])

                    pcs_mapping_val = pcs_mapping_contracts[all_evm_contracts_lists[bytecodes_lists_ids[ii]]]
                    pc_lists_val = []
                    for pc_val in all_evm_pc_lists[bytecodes_lists_ids[ii]]:
                        if pc_val in pcs_mapping_val:
                            pc_lists_val.append(pcs_mapping_val[pc_val])
                        else:
                            pc_lists_val.append({'begin': -1, 'end': -1})

                    pc_lists.append(pc_lists_val)

                # print(len(bytecodes_lists[0]))
                return bytecodes_lists, contract_names, opcodes_lists, pc_lists #10
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("compiled faild!")
                return None
    return None

def _get_positions_map(c_asm):  
    asm = c_asm['.data']['0'].copy()

    positions = asm['.code'].copy()
    while(True):
        try:
            positions.append(None)
            positions += asm['.data']['0']['.code'].copy()
            asm = asm['.data']['0'].copy()
        except:
            break

    pc_val = 0
    pc_position_map = {}
    for i in range(len(positions)):
        if positions[i]:
            pc_position_map[pc_val] = {'begin': positions[i]['begin'], 'end': positions[i]['end']}
            if ('value' in positions[i]) and (positions[i]['name'] != 'JUMP'):
                pc_val += 2
            else:
                pc_val += 1
    
    return pc_position_map

def make_bytecodes_opcodes_for_binary_file(file_path, instance_len, output_contract_dir, CFGBuilder_path):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()

        binruntime = file_content.strip()
    
        all_evm_bytecodes_lists = []
        all_evm_opcodes_lists =[]

        contract_names = ['BytecodeContract', '...']

        if '_' in binruntime:
            print(contract_names[0], "exist the third library!")
            # print(binruntime)
            pat = re.compile(r'__(<stdin>.*?)__')
            # pat = re.compile(r'__(.*[^_]?)_')
            res = list(set(pat.findall(binruntime)))
            print(res)
            replace_maps = {}
            for address_val in range(len(res)):
                replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
            binruntime = link_code(binruntime, replace_maps)

        contract_name = contract_names[0]
        file_dir = os.path.join(output_contract_dir, contract_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        compile_file_path = os.path.join(file_dir, contract_name + '.bin')
        with open(compile_file_path, 'w') as f:
            f.write(binruntime)

        cfg_file_path = os.path.join(file_dir, 'CFG.json')
        
        contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
        contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
        # contractInstance.buildCFG_dot_runtime()
        contractInstance.buildCFG_runtime()

        all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
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

def make_bytecodes_opcodes_for_opcode_file(file_path, instance_len, output_contract_dir, CFGBuilder_path):
    try:
        f = open(file_path)
        file_content = f.read()
        f.close()
        
        binruntime = translate_bytecodes_to_opcodes(file_content)
    
        all_evm_bytecodes_lists = []
        all_evm_opcodes_lists =[]

        contract_names = ['OpcodeContract', '...']

        if '_' in binruntime:
            print(contract_names[0], "exist the third library!")
            # print(binruntime)
            pat = re.compile(r'__(<stdin>.*?)__')
            # pat = re.compile(r'__(.*[^_]?)_')
            res = list(set(pat.findall(binruntime)))
            print(res)
            replace_maps = {}
            for address_val in range(len(res)):
                replace_maps[res[address_val]] = '0x{:040x}'.format(address_val + 1)
            binruntime = link_code(binruntime, replace_maps)

        contract_name = contract_names[0]
        file_dir = os.path.join(output_contract_dir, contract_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        compile_file_path = os.path.join(file_dir, contract_name + '.bin')
        with open(compile_file_path, 'w') as f:
            f.write(binruntime)

        cfg_file_path = os.path.join(file_dir, 'CFG.json')
        
        contractInstance = ContractInstance(CFGBuilder_path = CFGBuilder_path, output_dir = output_contract_dir)
        contractInstance.set_translate_path(runtimeBytecode_path = compile_file_path,  CFG_path = cfg_file_path)
        # contractInstance.buildCFG_dot_runtime()
        contractInstance.buildCFG_runtime()

        all_evm_bytecodes_lists_val, all_evm_opcodes_lists_val, all_evm_pc_lists_val = contractInstance.traverse_cfg_op()
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
    # make_bytecodes_continue2()

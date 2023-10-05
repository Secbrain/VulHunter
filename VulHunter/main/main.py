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
import argparse

from torch.optim.lr_scheduler import LambdaLR

import math
import torch.nn.functional as F

# import bytecodes_construction

import seaborn as sns
import matplotlib.pyplot as plt

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

vul_use_names = ['reentrancy-eth', 'controlled-array-length', 'suicidal', 'controlled-delegatecall', 'arbitrary-send', 'incorrect-equality', 'integer-overflow', 'unchecked-lowlevel', 'tx-origin', 'locked-ether', 'unchecked-send', 'costly-loop', 'erc721-interface', 'erc20-interface', 'timestamp', 'block-other-parameters', 'calls-loop', 'low-level-calls', 'erc20-indexed', 'erc20-throw', 'hardcoded', 'array-instead-bytes', 'unused-state', 'costly-operations-loop', 'external-function', 'send-transfer', 'boolean-equal', 'boolean-cst', 'uninitialized-state', 'tod']

def parse_args():  # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser(
        description="VulHunter. For usage information, run --help",
        usage="main.py contract.sol/bin/evm [flag]",
    )

    parser.add_argument("--contract", help="contract.sol")

    parser.add_argument(
        "--version",
        help="Displays the current version",
        version='0.1.0',
        action="version",
    )

    group_detector = parser.add_argument_group("Detectors")

    group_detector.add_argument(
        "--detectors",
        help="Comma-separated list of detectors, defaults to all, "
        "available detectors: {}".format(", ".join(d for d in vul_use_names)),
        action="store",
        dest="detectors",
        default='ALL'
    )

    group_detector.add_argument(
        "--list-detectors",
        help="List available detectors",
        action=ListDetectors,
        nargs=0,
        default=False,
    )

    group_misc = parser.add_argument_group("Miscs")

    group_misc.add_argument(
        "--filetype",
        help="Input the file type of the contract [solidity (default), bytecode, and opcode]",
        action="store",
        default='solidity',
    )

    group_misc.add_argument(
        "--solc-version",
        help="Input the solc version of compling contract 0.4.24 (default)",
        action="store",
        default='0.4.24',
    )

    group_misc.add_argument(
        "--ifmap",
        help="Input the retult mapping item: map  (default), nomap",
        action="store",
        default='map',
    )

    group_misc.add_argument(
        '--map-number',
        help='Set the number of mapping positions in instances (default 3)',
        action='store',
        default=3
    )

    group_misc.add_argument(
        "--instance-len",
        help="Input the number of extracting instances (default 10)",
        action="store",
        default=10,
    )

    group_misc.add_argument(
        '--report',
        help='Export the audit report as a pdf file ("--report -" to export to stdout)',
        action='store',
        default=None
    )
                        
    group_misc.add_argument(
        '--report-main',
        help='Export the main audit report as a pdf file ("--report-main -" to export to main stdout)',
        action='store',
        default=None
    )

    group_misc.add_argument(
        '--tmp-dir',
        help='Set the tmp dir (default .)',
        action='store',
        default='.'
    )

    group_misc.add_argument(
        '--model-dir',
        help='Set the model dir (default ./models)',
        action='store',
        default='./models'
    )

    group_train = parser.add_argument_group("Tranning")

    group_train.add_argument(
        "--train-contracts",
        help="Input the trainning contract files",
        action="store",
        default=''
    )

    group_train.add_argument(
        "--train-solcversions",
        help="Input the solc versions of contracts",
        action="store",
        default=''
    )

    group_train.add_argument(
        "--instance-dir",
        help="Input the dir path of generating contract instances",
        action="store",
        default=''
    )

    group_train.add_argument(
        "--train-labels",
        help="Input the trainning labels of contracts",
        action="store",
        default=''
    )

    group_train.add_argument(
        "--contract-instances",
        help="Input the trainning instances of contracts",
        action="store",
        default=''
    )

    group_train.add_argument(
        '--epoch',
        help='Set the trainning epoch (default 50)',
        action='store',
        default=50
    )

    group_train.add_argument(
        '--data-len',
        help='Set the instance length (default 512)',
        action='store',
        default=512
    )

    group_train.add_argument(
        '--batchsize',
        help='Set the batchsize (default 512)',
        action='store',
        default=512
    )

    group_verify = parser.add_argument_group("Verifying")

    group_verify.add_argument(
        '--verify',
        help='Set the batchsize (default False)',
        action='store_true',
        default=False
    )

    group_verify.add_argument(
        '--solver',
        help='Set the batchsize (default Z3), and select multiple solver by using commas to separate, e.g., Z3,Yices,CVC4. Also, this is equivalent to ALL.',
        action='store',
        default='Z3'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    return args

class ListDetectors(argparse.Action):  # pylint: disable=too-few-public-methods
    def __call__(self, parser, *args, **kwargs):  # pylint: disable=signature-differs
        print(vul_use_names)
        parser.exit()

def _convert_offset_to_line_column(pos, line_break_positions):
    ret = {}
    ret['begin'] = None
    ret['end'] = None
    if pos['begin'] >= 0 and (pos['end'] - pos['begin'] + 1) >= 0:
        ret['begin'] = _convert_from_char_pos(pos['begin'], line_break_positions)
        ret['end'] = _convert_from_char_pos(pos['end'], line_break_positions)
    return ret

def _convert_from_char_pos(pos, line_break_positions):
    line = _find_lower_bound(pos, line_break_positions)
    if len(line_break_positions) == 0 or line_break_positions[line] != pos:
        line += 1
    begin_col = 0 if line == 0 else line_break_positions[line - 1] + 1
    col = pos - begin_col
    return {'line': line, 'column': col}

def _find_lower_bound(target, array):
    start = 0
    length = len(array)
    while length > 0:
        half = length >> 1
        middle = start + half
        if array[middle] <= target:
            length = length - 1 - half
            start = middle + 1
        else:
            length = half
    return start - 1

def _load_line_break_positions(content):
    breakline_positions = []
    chanum = 0
    for line in content:
        if line[-1] == '\n':
            breakline_positions.append(chanum + len(line) - 1)
        chanum += len(line)
    return breakline_positions# [i for i, letter in enumerate(content) if letter == '\n']

if __name__ == '__main__':
    args = parse_args()

    # Extracting the instances of contracts based on the contract files and solc versions
    if args.train_contracts and args.train_solcversions and args.instance_dir:
        import bytecodes_construction

        print("Generating the contract instances...")
        print("train_contracts ", args.train_contracts)
        print("train_solcversions ", args.train_solcversions)
        print("instance_len ", args.instance_len)
        print("instance_dir ", args.instance_dir)
        
        bytecodes_construction.make_bytecodes_continue2(args.train_contracts, args.train_solcversions, int(args.instance_len), args.instance_dir)
        
        sys.exit()

    # Trainning the models based on the contract instances
    elif args.train_labels and args.contract_instances:
        import result_predict

        print("Trainning the detection models...")
        print("train_labels ", args.train_labels)
        print("contract_instances ", args.contract_instances)
        print("model_dir ", args.model_dir)

        vuldetects = []
        if args.detectors == 'ALL':
            vuldetects = vul_use_names
        else:
            for vul_val in args.detectors.split(','):
                if vul_val in vul_use_names:
                    vuldetects.append(vul_val)
                else:
                    print(vul_val + ' is not supported!')
        
        result_predict.train_vul_models(args.train_labels, args.contract_instances, vuldetects, args.model_dir, int(args.epoch), int(args.data_len), int(args.batchsize), int(args.instance_len))

        sys.exit()

    import bytecodes_construction
    import result_predict

    # Testing the contract
    filetype = args.filetype
    filepath = args.contract
    version =  None
    ifmap = None
    reporttype = None
    if filetype == 'solidity':
        version = args.solc_version
        ifmap = args.ifmap

    instance_len = int(args.instance_len)

    vuldetects = []
    if args.detectors == 'ALL':
        vuldetects = vul_use_names
    else:
        for vul_val in args.detectors.split(','):
            if vul_val in vul_use_names:
                vuldetects.append(vul_val)
            else:
                print(vul_val + ' is not supported!')

    print(filetype)
    print(filepath)
    print(version)
    print(ifmap)
    print(instance_len)

    time_begin = time.time()

    bytecodes_lists = []
    filecontent_lists = []
    contractsname_lists = []
    opcodes_lists = []
    pc_lists = []
    for i in range(1):
        # bytecodes_list_val, contractnames_val = bytecodes_construction.make_bytecodes_for_solidity_file(filepath, version)
        file = open(filepath,encoding='utf-8')
        file_content_val = file.readlines()

        if filetype == 'solidity' and ifmap == 'nomap':
            bytecodes_list_val, contractnames_val, opcodes_lists_val = bytecodes_construction.make_bytecodes_opcodes_for_solidity_file(filepath, version, instance_len)
        elif filetype == 'solidity' and ifmap == 'map':
            bytecodes_list_val, contractnames_val, opcodes_lists_val, pcs_lists_val = bytecodes_construction.make_bytecodes_opcodes_mapping_for_solidity_file(filepath, version, instance_len)
            pc_lists.append(pcs_lists_val)
        elif filetype == 'bytecode':
            bytecodes_list_val, contractnames_val, opcodes_lists_val = bytecodes_construction.make_bytecodes_opcodes_for_binary_file(filepath, instance_len)
        elif filetype == 'opcode':
            bytecodes_list_val, contractnames_val, opcodes_lists_val = bytecodes_construction.make_bytecodes_opcodes_for_opcode_file(filepath, instance_len)

        bytecodes_lists.append(bytecodes_list_val)
        opcodes_lists.append(opcodes_lists_val)
        filecontent_lists.append(file_content_val)
        contractsname_lists.append(contractnames_val)

    time_predict = time.time()
    audit_time = int((time_predict-time_begin)*1000)

    predict_results = result_predict.test_LSTM_prelabel(bytecodes_lists, vuldetects, args.model_dir, instance_len, int(args.data_len), int(args.batchsize))

    # print(predict_results)

    table_result = [['ID', 'Pattern', 'Description', 'Severity', 'Confidence', 'Status/Num', 'Description', 'Scenarios', 'Scenarios_supplement', 'Recommendation', 'Severity'], [1, 'reentrancy-eth', 'Re-entry vulnerabilities (Ethereum theft)', 'High', 'probably', 'Pass', 'A reentrancy error was detected. This is the reentry of ether. Through re-entry, the account balance can be maliciously withdrawn, resulting in losses. Do not report re-reporting that does not involve Ether (please refer to "reentrancy-no-eth")', 'function withdrawBalance(){<br/>&#160;&#160;&#160;&#160;// send userBalance[msg.sender] Ether to msg.sender<br/>&#160;&#160;&#160;&#160;// if mgs.sender is a contract, it will call its fallback function<br/>&#160;&#160;&#160;&#160;if( ! (msg.sender.call.value(userBalance[msg.sender])() ) ){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;throw;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;userBalance[msg.sender] = 0;<br/>}', 'Bob used the reentrance vulnerability to call `withdrawBalance` multiple times and withdrew more than he originally deposited into the contract.', 'Can adopt check-effects-interactions mode.', 'High'], [2, 'controlled-array-length', 'Length is allocated directly', 'High', 'probably', 'Pass', 'Detect direct allocation of array length.', 'contract A {<br/>&#160;&#160;&#160;&#160;uint[] testArray; // dynamic size array<br/>&#160;&#160;&#160;&#160;function f(uint usersCount) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;testArray.length = usersCount;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function g(uint userIndex, uint val) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;testArray[userIndex] = val;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Contract storage/state variables are indexed by 256-bit integers. Users can set the array length to 2 ** 256-1 to index all storage slots. In the above example, you can call function f to set the length of the array, and then call function g to control any storage slots needed. Please note that the storage slots here are indexed by the hash of the indexer. Nonetheless, all storage will still be accessible and can be controlled by an attacker.', 'It is not allowed to set the length of the array directly; instead, choose to add values as needed. Otherwise, please check the contract thoroughly to ensure that the user-controlled variables cannot reach the array length allocation.', 'High'], [3, 'suicidal', 'Check if anyone can break the contract', 'High', 'exactly', 'Pass', 'Due to lack of access control or insufficient access control, malicious parties can self-destruct the contract. Calling selfdestruct/suicide lacks protection.', 'contract Suicidal{<br/>&#160;&#160;&#160;&#160;function kill() public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;selfdestruct(msg.sender);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls the "kill" function and breaks the contract.', 'Protect access to all sensitive functions.', 'High'], [4, 'controlled-delegatecall', 'The delegate address out of control', 'High', 'probably', 'Pass', 'Delegate the call or call code to an address controlled by the user. The address of Delegatecall is not necessarily trusted, it is still a problem of access control, and the address is not checked.', 'contract Delegatecall{<br/>&#160;&#160;&#160;&#160;function delegate(address to, bytes data){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;to.delegatecall(data);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls `delegate` and delegates the execution of the malicious contract to him. As a result, Bob withdraws the funds from the contract and destroys the contract.', 'Avoid using `delegatecall`. If you use it, please only target trusted destinations.', 'High'], [5, 'arbitrary-send', 'Check if Ether can be sent to any address', 'High', 'probably', 'Pass', 'The call to the function that sends Ether to an arbitrary address has not been reviewed.', 'contract ArbitrarySend{<br/>&#160;&#160;&#160;&#160;address destination;<br/>&#160;&#160;&#160;&#160;function setDestination(){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination = msg.sender;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function withdraw() public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination.transfer(this.balance);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls setDestination and withdraw, and as a result, he withdraws the balance of the contract.', 'Ensure that no user can withdraw unauthorized funds.', 'High'], [6, 'tod', 'Transaction sequence dependence for receivers/ethers', 'High', 'probably', 'Pass', "Mainly about the receiver's exception. A person who is running an Ethereum node can tell which transactions are going to occur before they are finalized.A race condition vulnerability occurs when code depends on the order of the transactions submitted to it.", 'pragma solidity ^0.4.5;<br/>contract StandardToken is ERC20, BasicToken {<br/>&#160;&#160;&#160;&#160;...<br/>&#160;&#160;&#160;&#160;function approve(address _spender, uint256 _value) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;allowed[msg.sender][_spender] = _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;Approval(msg.sender, _spender, _value);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return true;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;...<br/>}', '', 'Before performing the approve change, reset the value to zero first, and then perform the change operation.', 'High'], [7, 'uninitialized-state', 'Check for uninitialized state variables', 'High', 'exactly', 'Pass', 'Uninitialized state variables can lead to intentional or unintentional vulnerabilities.', 'contract Uninitialized{<br/>&#160;&#160;&#160;&#160;address destination;<br/>&#160;&#160;&#160;&#160;function transfer() payable public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination.transfer(msg.value);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls "transfer". As a result, the ether is sent to the address "0x0" and is lost.', 'Initialize all variables. If you want to initialize a variable to zero, set it explicitly to zero.', 'High'], [8, 'parity-multisig-bug', 'Check for multi-signature vulnerabilities', 'High', 'probably', 'Pass', 'Multi-signature vulnerability. Hackers can use the initWallet function to call the initMultiowned function to obtain the identity of the contract owner.', 'contract WalletLibrary_bad is WalletEvents {<br/>&#160;&#160;&#160;&#160;function initWallet(address[] _owners, uint _required, uint _daylimit) { <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;initDaylimit(_daylimit); <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;initMultiowned(_owners, _required);<br/>&#160;&#160;&#160;&#160;}  // kills the contract sending everything to `_to`.<br/>&#160;&#160;&#160;&#160;function initMultiowned(address[] _owners, uint _required) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_numOwners = _owners.length + 1;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_owners[1] = uint(msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_ownerIndex[uint(msg.sender)] = 1; <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i = 0; i < _owners.length; ++i)<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_owners[2 + i] = uint(_owners[i]);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_ownerIndex[uint(_owners[i])] = 2 + i;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_required = _required;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'InitWallet, initDaylimit and initMultiowned add internal limited types to prohibit external calls: or if only_uninitMultiowned (m_numOwners) is detected in initMultiowned, no error will occur. The number of initializations can also be reviewed by judging m_numOwners.', 'High'], [9, 'incorrect-equality', 'Check the strict equality of danger', 'Medium', 'exactly', 'Pass', 'Using strict equality (== and !=), an attacker can easily manipulate these equality. Specifically: the opponent can forcefully send Ether to any address through selfdestruct() or through mining, thereby invalidating the strict judgment.', 'contract Crowdsale{<br/>&#160;&#160;&#160;&#160;function fund_reached() public returns(bool){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return this.balance == 100 ether;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Crowdsale relies on fund_reached to know when to stop the sale of tokens. Bob sends 0.1 ether. As a result, fund_reached is always false, and crowdsale is always true.', 'Do not use strict equality to determine whether an account has enough ether or tokens.', 'Medium'], [10, 'integer-overflow', 'Check for integer overflow', 'Medium', 'probably', 'Pass', 'When an arithmetic operation reaches the maximum or minimum size of the type, overflow/underflow will occur. For example, if a number is stored in the uint8 type, it means that the number is stored as an 8-bit unsigned number, ranging from 0 to 2^8-1. In computer programming, when an arithmetic operation attempts to create a value, an integer overflow occurs, and the value can be represented by a given number of bits-greater than the maximum value or less than the minimum value.', 'contract Intergeroverflow{<br/>&#160;&#160;&#160;&#160;function bad() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint a;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint b;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint c = a + b;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Use Safemath to perform integer arithmetic or verify calculated values.', 'Medium'], [11, 'unchecked-lowlevel', 'Check for uncensored low-level calls', 'Medium', 'probably', 'Pass', 'The low-level call to the external contract failed, and the return value was not judged. When sending ether at the same time, please check the return value and handle the error.', 'contract MyConc{<br/>&#160;&#160;&#160;&#160;function my_func(address payable dst) public payable{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;dst.call.value(msg.value)("");<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The return value of the low-level call is not checked, so if the call fails, the ether will be locked in the contract. If you use low-level calls to block block operations, consider logging the failed calls.', 'Make sure to check or record the return value of low-level calls.', 'Medium'], [12, 'tx-origin', 'Check the dangerous use of tx.origin', 'Medium', 'probably', 'Pass', 'If a legitimate user interacts with a malicious contract, the protection based on tx.origin will be abused by the malicious contract.', 'contract TxOrigin {<br/>&#160;&#160;&#160;&#160;address owner = msg.sender;<br/>&#160;&#160;&#160;&#160;function bug() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(tx.origin == owner);<br/>&#160;&#160;&#160;&#160;}<br/>}', "Bob is the owner of TxOrigin. Bob calls Eve's contract. Eve's contract is called TxOrigin and bypasses the protection of tx.origin.", 'Do not use `tx.origin` for authorization.', 'Medium'], [13, 'locked-ether', 'Whether the contract ether is locked', 'Medium', 'exactly', 'Pass', 'A contract programmed to receive ether (with the payable logo) should implement the method of withdrawing ether, that is, call transfer (recommended), send or call.value at least once.', 'pragma solidity 0.4.24;<br/>contract Locked{<br/>&#160;&#160;&#160;&#160;function receive() payable public{}<br/>}', 'All Ether sent to "Locked" will be lost.', 'Delete payable attributes or add withdrawal functions.', 'Medium'], [14, 'unchecked-send', 'Check unreviewed send', 'Medium', 'probably', 'Pass', 'Similar to unchecked-lowlevel, it is explained here that the return value of send and Highlevelcall is not checked.', 'contract MyConc{<br/>&#160;&#160;&#160;&#160;function my_func(address payable dst) public payable{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;dst.send(msg.value);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The return value of send is not checked, so if the send fails, the ether will be locked in the contract. If you use send to prevent block operations, please consider logging failed send.', 'Make sure to check or record the return value of send.', 'Medium'], [15, 'boolean-cst', 'Check for misuse of Boolean constants', 'Medium', 'probably', 'Pass', 'Detect abuse of Boolean constants. Bool variable is used incorrectly, here is the operation of bool variable.', 'contract A {<br/>&#160;&#160;&#160;&#160;function f(uint x) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (false) { // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function g(bool b) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return (b || true); // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The Boolean constants in the code have very few legal uses. Other uses (as conditions in complex expressions) indicate the persistence of errors or error codes.', 'Verify and simplify the conditions.', 'Medium'], [16, 'erc721-interface', 'Check the wrong ERC721 interface', 'Medium', 'exactly', 'Pass', 'The return value of the "ERC721" function is incorrect. Interacting with these functions, the contract of solidity version> 0.4.22 will not be executed because of the lack of return value.', 'contract Token{<br/>&#160;&#160;&#160;&#160;function ownerOf(uint256 _tokenId) external view returns (bool);<br/>&#160;&#160;&#160;&#160;//...<br/>}', "Token.ownerOf does not return the expected boolean value. Bob deploys the token. Alice creates a contract to interact with, but uses the correct `ERC721` interface implementation. Alice's contract cannot interact with Bob's contract.", 'Set appropriate return values and vtypes for the defined ʻERC721` function.', 'Medium'], [17, 'erc20-interface', 'Check for wrong ERC20 interface', 'Medium', 'exactly', 'Pass', 'The return value of the "ERC20" function is incorrect. Interacting with these functions, the contract of solidity version> 0.4.22 will not be executed because of the lack of return value.', 'contract Token{<br/>&#160;&#160;&#160;&#160;function transfer(address to, uint value) external;<br/>&#160;&#160;&#160;&#160;//...<br/>}', "Token.transfer does not return the expected boolean value. Bob deploys the token. Alice creates a contract to interact with, but uses the correct `ERC20` interface implementation. Alice's contract cannot interact with Bob's contract.", 'Set appropriate return values and vtypes for the defined ʻERC20` function.', 'Medium'], [18, 'costly-loop', 'Check for too expensive loops', 'Low', 'possibly', 'Pass', 'Ethereum is a very resource-constrained environment. The price of each calculation step is several orders of magnitude higher than the price of the centralized provider. In addition, Ethereum miners impose limits on the total amount of natural gas consumed in the block. If array.length is large enough, the function exceeds the gas limit, and the transaction that calls the function will never be confirmed. If external participants influence array.length, this will become a security issue.', 'pragma solidity 0.4.24;<br/>contract PriceOracle {<br/>&#160;&#160;&#160;&#160;address internal owner;<br/>&#160;&#160;&#160;&#160;address[] public subscribers;<br/>&#160;&#160;&#160;&#160;mapping(address => uint) balances;<br/>&#160;&#160;&#160;&#160;uint internal constant PRICE = 10**15;<br/>&#160;&#160;&#160;&#160;function subscribe() payable external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;subscribers.push(msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[msg.sender] += msg.value;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function setPrice(uint price) external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(msg.sender == owner);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;bytes memory data = abi.encodeWithSelector(SIGNATURE, price);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i = 0; i < subscribers.length; i++) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if(balances[subscribers[i]] >= PRICE) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[subscribers[i]] -= PRICE;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;subscribers[i].call.gas(50000)(data);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Please check the dynamic array of the loop carefully. If you find it can be exploited by an attacker, please change it to prevent the contract from executing too many loops and causing gas overflow and rollback.', 'Low'], [19, 'timestamp', 'The dangerous use of block.timestamp', 'Low', 'probably', 'Pass', 'There is a strict comparison with block.timestamp or now in the contract, and miners can benefit from block.timestamp.', 'contract Timestamp{<br/>&#160;&#160;&#160;&#160;event Time(uint);<br/>&#160;&#160;&#160;&#160;modifier onlyOwner {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.timestamp == 0);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;_;  <br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function bad0() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.timestamp == 0);<br/>&#160;&#160;&#160;&#160;}<br/>}', "Bob's contract relies on the randomness of block.timestamp. Eve is a miner who manipulates block.timestamp to take advantage of Bob's contract.", 'Avoid relying on block.timestamp.', 'Low'], [20, 'block-other-parameters', 'Hazardous use variables (block.number etc.)', 'Low', 'probably', 'Pass', 'Contracts usually require access to time values \u200b\u200bto perform certain types of functions. block.number can let you know the current time or time increment, but in most cases it is not safe to use them. block.number The block time of Ethereum is usually about 14 seconds, so the time increment between blocks can be predicted. However, the lockout time is not fixed and may change due to various reasons (for example, fork reorganization and difficulty coefficient). Since the block time is variable, block.number should not rely on accurate time calculations. The ability to generate random numbers is very useful in various applications. An obvious example is a gambling DApp, where a pseudo-random number generator is used to select the winner. However, creating a sufficiently powerful source of randomness in Ethereum is very challenging. Using blockhash, block.difficulty and other areas is also unsafe because they are controlled by miners. If the stakes are high, the miner can mine a large number of blocks by renting hardware in a short period of time, select the block that needs to obtain the block hash value to win, and then discard all other blocks.', 'contract Otherparameters{<br/>&#160;&#160;&#160;&#160;event Number(uint);<br/>&#160;&#160;&#160;&#160;event Coinbase(address);<br/>&#160;&#160;&#160;&#160;event Difficulty(uint);<br/>&#160;&#160;&#160;&#160;event Gaslimit(uint);<br/>&#160;&#160;&#160;&#160;function bad0() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.number == 20);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.coinbase == msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.difficulty == 20);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.gaslimit == 20);<br/>&#160;&#160;&#160;&#160;}<br/>}', "The randomness of Bob's contract depends on block.number and so on. Eve is a miner who manipulates block.number and so on to use Bob's contract.", 'Avoid relying on block.number and other data that can be manipulated by miners.', 'Low'], [21, 'calls-loop', 'Check the external call in the loop', 'Low', 'probably', 'Pass', 'Check that the key access control ETH is transmitted cyclically. If at least one address cannot receive ETH (for example, it is a contract with a default fallback function), the entire transaction will be restored. Loss of parameters.', 'contract CallsInLoop{<br/>&#160;&#160;&#160;&#160;address[] destinations;<br/>&#160;&#160;&#160;&#160;constructor(address[] newDestinations) public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destinations = newDestinations;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function bad() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i=0; i < destinations.length; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destinations[i].transfer(i);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', 'If one of the destination addresses is restored by the rollback function, bad() will restore all, so all the work is wasted.', 'Try to avoid calling external contracts in the loop, and you can use the pull over push strategy.', 'Low'], [22, 'low-level-calls', 'Check low-level calls', 'Info', 'exactly', 'Pass', 'Label low-level methods such as call, delegatecall, and callcode, because these methods are easily exploited by attackers.', 'contract Sender {<br/>&#160;&#160;&#160;&#160;address owner;<br/>&#160;&#160;&#160;&#160;modifier onlyceshi() {<br/>&#160;&#160;&#160;&#160;owner.callcode(bytes4(keccak256("inc()")));<br/>&#160;&#160;&#160;&#160;_;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function send(address _receiver) payable external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;_receiver.call.value(msg.value).gas(7777)("");<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function sendceshi(address _receiver) payable external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if(_receiver.call.value(msg.value).gas(7777)("")){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;revert();<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Avoid low-level calls. Check whether the call is successful. If the call is to sign a contract, please check whether the code exists.', 'Informational'], [23, 'erc20-indexed', 'ERC20 event parameter is missing indexed', 'Info', 'exactly', 'Pass', 'The address parameters of the "Transfer" and "Approval" events of the ERC-20 token standard shall include indexed.', 'contract ERC20Bad {<br/>&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;event Transfer(address from, address to, uint value);<br/>&#160;&#160;&#160;&#160;event Approval(address owner, address spender, uint value);<br/>&#160;&#160;&#160;&#160;// ...<br/>}', 'According to the definition of the ERC20 specification, the first two parameters of the Transfer and Approval events should carry the indexed keyword. If these keywords are not included, the parameter data will be excluded from the bloom filter of the transaction/block. Therefore, external tools searching for these parameters may ignore them and fail to index the logs in this token contract.', 'According to the ERC20 specification, the indexed keyword is added to the event parameter of the corresponding keyword.', 'Informational'], [24, 'erc20-throw', 'ERC20 throws an exception', 'Info', 'exactly', 'Pass', 'The function of the ERC-20 token standard should be thrown in the following special circumstances: if there are not enough tokens in the _from account balance to spend, it should be thrown; unless the _from account deliberately authorizes the sending of messages through some mechanism Otherwise, transferFrom should be thrown.', 'contract SomeToken {<br/>&#160;&#160;&#160;&#160;mapping(address => uint256) balances;<br/>&#160;&#160;&#160;&#160;event Transfer(address indexed _from, address indexed _to, uint256 _value);<br/>&#160;&#160;&#160;&#160;function transfer(address _to, uint _value) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (_value > balances[msg.sender] || _value > balances[_to] + _value) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return false;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[msg.sender] = balances[msg.sender] - _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[_to] = balances[_to] + _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;emit Transfer(msg.sender, _to, _value);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return true;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Add the corresponding throw method to ERC-20 tokens.', 'Informational'], [25, 'hardcoded', 'Check the legitimacy of the address', 'Info', 'probably', 'Pass', 'The contract contains an unknown address, which may be used for some malicious activities. Need to check the hard-coded address and its purpose. The address length is prone to errors, and the length of the address is not enough, it will not report an error, so it is very dangerous to write a mistake. Here is an identification.', 'contract C {<br/>&#160;&#160;&#160;&#160;function f(uint a, uint b) pure returns (address) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;address public multisig = 0xf64B584972FE6055a770477670208d737Fff282f;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return multisig;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Check carefully whether the address is wrong, and if there is an error, please take the time to correct it.', 'Informational'], [26, 'array-instead-bytes', 'The byte array can be replaced with bytes', 'Opt', 'exactly', 'Pass', 'byte[] can be converted to bytes to save gas resources.', 'pragma solidity 0.4.24;<br/>contract C {<br/>&#160;&#160;&#160;&#160;byte[] someVariable;<br/>&#160;&#160;&#160;&#160;...<br/>}', '', 'Replacing byte[] with bytes can save gas.', 'Optimization'], [27, 'unused-state', 'Check unused state variables', 'Opt', 'exactly', 'Pass', 'Unused variables are allowed in Solidity, and they do not pose direct security issues. The best practice is to avoid them as much as possible: resulting in increased calculations (and unnecessary gas consumption) means errors or incorrect data structures, and usually means poor code quality leads to code noise and reduces code readability.', 'contract A{<br/>&#160;&#160;&#160;&#160;address unused;<br/>&#160;&#160;&#160;&#160;address public unused2;<br/>&#160;&#160;&#160;&#160;address private unused3;<br/>&#160;&#160;&#160;&#160;address unused4;<br/>&#160;&#160;&#160;&#160;address used;<br/>&#160;&#160;&#160;&#160;function ceshi1 () external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;unused3 = address(0);<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Delete unused state variables.', 'Optimization'], [28, 'costly-operations-loop', 'Expensive operations in the loop', 'Opt', 'probably', 'Pass', 'Expensive operations within the loop.', 'contract CostlyOperationsInLoop{<br/>&#160;&#160;&#160;&#160;uint loop_count = 100;<br/>&#160;&#160;&#160;&#160;uint state_variable=0;<br/>&#160;&#160;&#160;&#160;function bad() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i=0; i < loop_count; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;state_variable++;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function good() external{<br/>&#160;&#160;&#160;&#160;  uint local_variable = state_variable;<br/>&#160;&#160;&#160;&#160;  for (uint i=0; i < loop_count; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;local_variable++;<br/>&#160;&#160;&#160;&#160;  }<br/>&#160;&#160;&#160;&#160;  state_variable = local_variable;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Due to the expensive SSTOREs, the incremental state variables in the loop will generate a large amount of gas, which may result in insufficient gas.', 'This test is a state variable in the loop. The state variable costs more gas than the local variable. This was not tested before, and only length was tested before.', 'Optimization'], [29, 'send-transfer', 'Check Transfe to replace Send', 'Opt', 'exactly', 'Pass', 'The recommended way to perform the check of Ether payment is addr.transfer(x). If the transfer fails, an exception is automatically raised.', 'if(!addr.send(42 ether)) {<br/>&#160;&#160;&#160;&#160;revert();<br/>}', '', 'It is safer to use transfer instead of send.', 'Optimization'], [30, 'boolean-equal', 'Check comparison with boolean constant', 'Opt', 'exactly', 'Pass', "Check the comparison of Boolean constants. There is no need to compare with true and false, so it's superfluous (gas consumption).", 'contract A {<br/>&#160;&#160;&#160;&#160;function f(bool x) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (x == true) { // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;   // ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Boolean constants can be used directly without comparison with true or false.', 'Delete the equation equal to the Boolean constant.', 'Optimization'], [31, 'external-function', 'Public functions can be declared as external', 'Opt', 'exactly', 'Pass', 'Functions with public visibility modifiers are not called internally. Changing the visibility level to an external level can improve the readability of the code. In addition, in many cases, functions that use external visibility modifiers cost less gas than functions that use public visibility modifiers.', 'contract ContractWithFunctionCalledSuper {<br/>&#160;&#160;&#160;&#160;function callWithSuper() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint256 i = 0;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The callWithSuper() function can be declared as external visibility.', 'Use the "external" attribute for functions that are never called from within the contract.', 'Optimization']]

#     map_severity = {}
#     for i in range(1, len(table_result)):
#         map_severity[table_result[i][1]] = (table_result[i][3], table_result[i][4], table_result[i][2])

    # audit_time = 0
    time_start = time.localtime()
    auditid = ""
    auditid = auditid + time.strftime("%m%d%H%M%S", time_start)
    auditid = auditid + str(random.randint(1000,9999))    
    filename = filepath.split('/')[-1]

    tmp_dir = args.tmp_dir

    line_break_positions = None
    tmpfig_path = os.path.join(tmp_dir, auditid)
    if ifmap == 'map':
        filecontent_val = filecontent_lists[0]
        line_break_positions = _load_line_break_positions(filecontent_val)
        if not os.path.exists(tmpfig_path):
            os.makedirs(tmpfig_path)

    positions_results = {}

    solvers = None
    if args.verify:
        from feasibility_verifier import validate_feasibility_instance
        solvers = args.solver.split(',')

    holeloops = []
    max_n = int(args.map_number)
    for i in range(1, len(table_result)):
        if table_result[i][1] not in predict_results:
            continue
        predict_labels = predict_results[table_result[i][1]]['predict_labels']
        instances_predict_labels = predict_results[table_result[i][1]]['instances_predict_labels']
        instances_predict_weights = predict_results[table_result[i][1]]['instances_predict_weights']

        for ii in range(len(predict_labels)):
            if predict_labels[ii] == 1:
                print("Vulnerability: {}, Severity: {}, Confidence: {}, Description: {}, instance: {}".format(table_result[i][1], table_result[i][3], table_result[i][4], table_result[i][2], instances_predict_labels[ii]))

                if ifmap == 'map':
                    plt.figure(figsize=(7.5, 6), dpi=800)
                    sns.heatmap(instances_predict_weights[ii], vmin=0.0, vmax=1.0, cmap="Blues")
                    plt.yticks([])
                    png_path = os.path.join(tmpfig_path, table_result[i][1] + '.png')
                    aaa=plt.savefig(png_path)
                    # plt.show()
                    plt.close()

                    positions_all = []

                    feasible_flag = False

                    for iii in range(len( instances_predict_labels[ii])):
                        if  instances_predict_labels[ii][iii] == 1:
                            if args.verify:
                                verify_result_val = validate_feasibility_instance(opcodes_lists[0][iii], solvers)
                                feasible_flag = feasible_flag | verify_result_val
                                print("[Path validation] The {}-th instance is suspected vulnerability sequence!: {}.".format(iii+1, verify_result_val))
                                if not verify_result_val:
                                    predict_results[table_result[i][1]]['instances_predict_labels'][ii][iii] = 0 # correct false predictions
                                    continue

                            print("The {}-th instance is suspected vulnerability sequence!".format(iii+1))
                            sequence_weights = np.array(instances_predict_weights[ii][iii])
                            # import pdb
                            # pdb.set_trace()
                            inds = np.argpartition(sequence_weights, -max_n)[-max_n:]
                            inds = inds[np.argsort(-sequence_weights[inds])]

                            positions_val = []
                            for iiii in range(max_n):
                                if len(pc_lists[ii][iii]) <= inds[iiii]:
                                    print(" - The {}-th position of suspected source code (importance: {}): {}.".format(iiii+1, sequence_weights[inds[iiii]], "filled zeros"))
                                    positions_val.append((sequence_weights[inds[iiii]], False, "filled zeros", inds[iiii]))
                                elif pc_lists[ii][iii][inds[iiii]]['begin'] == -1 and pc_lists[ii][iii][inds[iiii]]['end'] == -1:
                                    print(" - The {}-th position of suspected source code (importance: {}): {}.".format(iiii+1, sequence_weights[inds[iiii]], "auxdata"))
                                    positions_val.append((sequence_weights[inds[iiii]], False, "auxdata", inds[iiii]))
                                else:
                                    position_val = _convert_offset_to_line_column(pc_lists[ii][iii][inds[iiii]], line_break_positions)
                                    print(" - The {}-th position of suspected source code (importance: {}): {}-{}.".format(iiii+1, sequence_weights[inds[iiii]], position_val['begin'], position_val['end']))
                                    positions_val.append((sequence_weights[inds[iiii]], True, position_val, inds[iiii]))
                            positions_all.append((iii, positions_val))
                    
                    if args.verify:
                        if feasible_flag:
                            holeloops.append({'title':table_result[i][1], 'severity': table_result[i][3], 'confidence': table_result[i][4], 'instances': instances_predict_labels[ii], 'description': table_result[i][2], 'positions': positions_all})
                        else:
                            predict_results[table_result[i][1]]['predict_labels'][ii] = 0
                    else:
                        holeloops.append({'title':table_result[i][1], 'severity': table_result[i][3], 'confidence': table_result[i][4], 'instances': instances_predict_labels[ii], 'description': table_result[i][2], 'positions': positions_all})

                    if args.report or args.report_main:
                        positions_results[table_result[i][1]] = (png_path, positions_all)
                else:
                    if args.verify:
                        feasible_flag = False
                        for iii in range(len( instances_predict_labels[ii])):
                            if  instances_predict_labels[ii][iii] == 1:
                                verify_result_val = validate_feasibility_instance(opcodes_lists[0][iii], solvers)
                                feasible_flag = feasible_flag | verify_result_val
                                print("[Path validation] The {}-th instance is suspected vulnerability sequence!: {}.".format(iii+1, verify_result_val))
                                if not verify_result_val: # correct false predictions
                                    predict_results[table_result[i][1]]['instances_predict_labels'][ii][iii] = 0
                        if feasible_flag:
                            holeloops.append({'title':table_result[i][1], 'severity': table_result[i][3], 'confidence': table_result[i][4], 'instances': instances_predict_labels[ii], 'description': table_result[i][2]})
                        else:
                            predict_results[table_result[i][1]]['predict_labels'][ii] = 0
                    else:
                        holeloops.append({'title':table_result[i][1], 'severity': table_result[i][3], 'confidence': table_result[i][4], 'instances': instances_predict_labels[ii], 'description': table_result[i][2]})

                # if args.verify:
                #     for iii in range(len( instances_predict_labels[ii])):
                #         if  instances_predict_labels[ii][iii] == 1:
                #             print("[Path validation] The {}-th instance is suspected vulnerability sequence!: {}.".format(iii+1, validate_feasibility_instance(opcodes_lists[0][iii], solvers)))

#         description_vals.append('[' + ','.join([str(vv) for vv in result_predict]) + ']')
    print("The extracted instances of the contract")
    print(bytecodes_lists[0])
    print(opcodes_lists[0])
#     print("time overhead: {} {}, all: {}".format(time_instance-time_begin, time_predict-time_instance, time_predict-time_begin))

    print('audittime:' + str(audit_time))
    # print(holeloops)
    holeloops_json = json.dumps(holeloops, cls=NpEncoder)
    print(holeloops_json)

    print("The information of report generation:")

    auditcontent = ""
    lines = filecontent_lists[0]
    for i in range(len(lines)-1):
        if lines[i]:
            # print(lines[i])
            auditcontent = auditcontent + lines[i].replace("\t","&#160;&#160;&#160;&#160;").replace(" ","&#160;").replace("\n","<br/>")
    if lines[len(lines)-1]:
        # print(lines[len(lines)-1])
        auditcontent = auditcontent + lines[len(lines)-1].replace("\t","&#160;&#160;&#160;&#160;").replace(" ","&#160;").replace("\n","")

    contracts_names = contractsname_lists[0]

    bytecodes = bytecodes_lists[0]
    opcodes = opcodes_lists[0]

#     print(contracts_names)

    if len(filename) > 30:
        filename = filename[:20] + "..." + filename[-10:]

    if args.report or args.report_main:
        from report_english import ReportEnglish

    if args.report:
        rep = ReportEnglish()
        rep._output(predict_results, filename, time_start, auditcontent, args.report, contracts_names, auditid, bytecodes, opcodes, ifmap, positions_results, filecontent_lists[0])
    if args.report_main:
        rep = ReportEnglish()
        rep._output_main(predict_results, filename, time_start, auditcontent, args.report_main, contracts_names, auditid, bytecodes, opcodes, ifmap, positions_results, filecontent_lists[0])


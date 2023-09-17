#!/usr/bin/env python

# EVM disassembler
from manticore.platforms.evm import *
from manticore.core.smtlib import *
from manticore.core.smtlib.visitors import *
from manticore.utils import log

from manticore.core.smtlib.solver import (
    Z3Solver,
    YicesSolver,
    CVC4Solver,
    BoolectorSolver,
    PortfolioSolver,
)
from manticore.native.memory import *
from manticore.platforms.evm import EVMWorld, ConcretizeArgument, concretized_args, Return, Stop

import pyevmasm
from pyevmasm import disassemble_hex
from pyevmasm import assemble_hex

# log.set_verbosity(9)
config.out_of_gas = 1

def translate_bytecodes_to_opcodes(opcodes_val):
    print(opcodes_val)
    map_replace = {'KECCAK256': 'SHA3', 'PC': 'GETPC', 'INVALID': 'ASSERTFAIL', 'SELFDESTRUCT': 'SUICIDE'}

    i = 0
    opcodes_val_length = len(opcodes_val)
    opcodes_str = ""
    bytecode_val = "0x"
    while i <  opcodes_val_length:
        try:
            if opcodes_val[i] == '':
                i = i + 1
            else:
                if opcodes_val[i][:2] == '0x':
                    opcodes_str = opcodes_str + opcodes_val[i] + '\n'
                elif opcodes_val[i] in ['KECCAK256', 'PC']:
                    opcode_val_str = map_replace[opcodes_val[i]]
                    hex_val = assemble_hex(opcode_val_str)
                    opcodes_str = opcodes_str + opcode_val_str + '\n'
                else:
                    hex_val = assemble_hex(opcodes_val[i])
                    opcodes_str = opcodes_str + opcodes_val[i] + '\n'
                i = i + 1
        except:
            if opcodes_val[i][:4] == 'PUSH' and (i + 2) < opcodes_val_length and opcodes_val[i + 2] in ['JUMP', 'JUMPI', 'JUMPDEST']:
                if (i + 3) < opcodes_val_length and opcodes_val[i + 3] in ['JUMPDEST', 'JUMP', 'JUMPI']:
                    i = i + 4
                else:
                    i = i + 3
            else:
                opcodes_str = opcodes_str + opcodes_val[i] + ' ' + opcodes_val[i+1] + '\n'
                i = i + 2
    return opcodes_str.strip()

def printi(instruction):
    print(f"Instruction: {instruction}")
    print(f"\tdescription: {instruction.description}")
    print(f"\tgroup: {instruction.group}")
    print(f"\taddress: {instruction.offset}")
    print(f"\tsize: {instruction.size}")
    print(f"\thas_operand: {instruction.has_operand}")
    print(f"\toperand_size: {instruction.operand_size}")
    print(f"\toperand: {instruction.operand}")
    print(f"\tsemantics: {instruction.semantics}")
    print(f"\tpops: {instruction.pops}")
    print(f"\tpushes:", instruction.pushes)
    print(f"\tbytes: 0x{instruction.bytes.hex()}")
    print(f"\twrites to stack: {instruction.writes_to_stack}")
    print(f"\treads from stack: {instruction.reads_from_stack}")
    print(f"\twrites to memory: {instruction.writes_to_memory}")
    print(f"\treads from memory: {instruction.reads_from_memory}")
    print(f"\twrites to storage: {instruction.writes_to_storage}")
    print(f"\treads from storage: {instruction.reads_from_storage}")
    print(f"\tis terminator {instruction.is_terminator}")


constraints = ConstraintSet()

# code = EVMAsm.assemble(
#     """
#     INVALID
#     """
#  )

opcodes_list = ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0xaf', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x18dc53e8', 'EQ', 'PUSH2', '0xb4', 'JUMPI', 'DUP1', 'PUSH4', '0x41c0e1b5', 'EQ', 'PUSH2', '0xea', 'JUMPI', 'DUP1', 'PUSH4', '0x721b0f22', 'EQ', 'PUSH2', '0x101', 'JUMPI', 'DUP1', 'PUSH4', '0x83197ef0', 'EQ', 'PUSH2', '0x137', 'JUMPI', 'DUP1', 'PUSH4', '0x8777366e', 'EQ', 'PUSH2', '0x14e', 'JUMPI', 'DUP1', 'PUSH4', '0x90b77c2e', 'EQ', 'PUSH2', '0x184', 'JUMPI', 'DUP1', 'PUSH4', '0x9c63fd56', 'EQ', 'PUSH2', '0x1ba', 'JUMPI', 'DUP1', 'PUSH4', '0xa2297f60', 'EQ', 'PUSH2', '0x1f0', 'JUMPI', 'DUP1', 'PUSH4', '0xbff0d811', 'EQ', 'PUSH2', '0x23e', 'JUMPI', 'DUP1', 'PUSH4', '0xed8f9838', 'EQ', 'PUSH2', '0x274', 'JUMPI', 'JUMPDEST', 'PUSH2', '0x2a8', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH2', '0x5c9', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'PUSH2', '0x60e', 'JUMPI', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'CALLVALUE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'ISZERO', 'PUSH2', '0x60d', 'JUMPI', 'INVALID']
#['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0xaf', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x18dc53e8', 'EQ', 'PUSH2', '0xb4', 'JUMPI', 'DUP1', 'PUSH4', '0x41c0e1b5', 'EQ', 'PUSH2', '0xea', 'JUMPI', 'DUP1', 'PUSH4', '0x721b0f22', 'EQ', 'PUSH2', '0x101', 'JUMPI', 'DUP1', 'PUSH4', '0x83197ef0', 'EQ', 'PUSH2', '0x137', 'JUMPI', 'DUP1', 'PUSH4', '0x8777366e', 'EQ', 'PUSH2', '0x14e', 'JUMPI', 'DUP1', 'PUSH4', '0x90b77c2e', 'EQ', 'PUSH2', '0x184', 'JUMPI', 'DUP1', 'PUSH4', '0x9c63fd56', 'EQ', 'PUSH2', '0x1ba', 'JUMPI', 'DUP1', 'PUSH4', '0xa2297f60', 'EQ', 'PUSH2', '0x1f0', 'JUMPI', 'DUP1', 'PUSH4', '0xbff0d811', 'EQ', 'PUSH2', '0x23e', 'JUMPI', 'DUP1', 'PUSH4', '0xed8f9838', 'EQ', 'PUSH2', '0x274', 'JUMPI', 'JUMPDEST', 'PUSH2', '0x2a8', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH2', '0x5c9', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'PUSH2', '0x60e', 'JUMPI', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'CALLVALUE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'ISZERO', 'PUSH2', '0x60d', 'JUMPI', 'JUMPDEST', 'JUMPDEST', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0xaf', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x18dc53e8', 'EQ', 'PUSH2', '0xb4', 'JUMPI', 'DUP1', 'PUSH4', '0x41c0e1b5', 'EQ', 'PUSH2', '0xea', 'JUMPI', 'DUP1', 'PUSH4', '0x721b0f22', 'EQ', 'PUSH2', '0x101', 'JUMPI', 'DUP1', 'PUSH4', '0x83197ef0', 'EQ', 'PUSH2', '0x137', 'JUMPI', 'DUP1', 'PUSH4', '0x8777366e', 'EQ', 'PUSH2', '0x14e', 'JUMPI', 'DUP1', 'PUSH4', '0x90b77c2e', 'EQ', 'PUSH2', '0x184', 'JUMPI', 'DUP1', 'PUSH4', '0x9c63fd56', 'EQ', 'PUSH2', '0x1ba', 'JUMPI', 'DUP1', 'PUSH4', '0xa2297f60', 'EQ', 'PUSH2', '0x1f0', 'JUMPI', 'DUP1', 'PUSH4', '0xbff0d811', 'EQ', 'PUSH2', '0x23e', 'JUMPI', 'DUP1', 'PUSH4', '0xed8f9838', 'EQ', 'PUSH2', '0x274', 'JUMPI', 'JUMPDEST', 'PUSH2', '0x2a8', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH2', '0x5c9', 'JUMP', 'JUMPDEST', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'CALLVALUE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'ISZERO', 'PUSH2', '0x606', 'JUMPI', 'INVALID']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xba4de90e', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x76', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH1', '0x78', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'ISZERO', 'PUSH1', '0x84', 'JUMPI', 'JUMPDEST', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'CALLVALUE', 'PUSH2', '0x1e61', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'POP', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'PUSH1', '0xc4', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xba4de90e', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x76', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH1', '0x78', 'JUMP', 'JUMPDEST', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'CALLVALUE', 'PUSH2', '0x1e61', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'POP', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'PUSH1', '0xb8', 'JUMPI', 'JUMPDEST', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xba4de90e', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x76', 'PUSH1', '0x4', 'DUP1', 'CALLDATASIZE', 'SUB', 'DUP2', 'ADD', 'SWAP1', 'DUP1', 'DUP1', 'CALLDATALOAD', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'SWAP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'SWAP3', 'SWAP2', 'SWAP1', 'POP', 'POP', 'POP', 'PUSH1', '0x78', 'JUMP', 'JUMPDEST', 'DUP1', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'CALLVALUE', 'PUSH2', '0x1e61', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'POP', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'ISZERO', 'PUSH1', '0xb8', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0x41', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x8aeccda8', 'EQ', 'PUSH2', '0x46', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH2', '0x52', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH2', '0x5b', 'PUSH2', '0x5d', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'ISZERO', 'PUSH2', '0x67', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'NUMBER', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0x76', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'COINBASE', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xb0', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'DIFFICULTY', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xbf', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0x41', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x8aeccda8', 'EQ', 'PUSH2', '0x46', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH2', '0x52', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH2', '0x5b', 'PUSH2', '0x5d', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'ISZERO', 'PUSH2', '0x67', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'NUMBER', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0x76', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'COINBASE', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xb0', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'DIFFICULTY', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xbf', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'GASLIMIT', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xce', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0x41', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x8aeccda8', 'EQ', 'PUSH2', '0x46', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH2', '0x52', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH2', '0x5b', 'PUSH2', '0x5d', 'JUMP', 'JUMPDEST', 'PUSH1', '0x14', 'NUMBER', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0x6c', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'COINBASE', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xa6', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'DIFFICULTY', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xb5', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0x41', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x8aeccda8', 'EQ', 'PUSH2', '0x46', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH2', '0x52', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH2', '0x5b', 'PUSH2', '0x5d', 'JUMP', 'JUMPDEST', 'PUSH1', '0x14', 'NUMBER', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0x6c', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'COINBASE', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xa6', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'DIFFICULTY', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xb5', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x14', 'GASLIMIT', 'EQ', 'ISZERO', 'ISZERO', 'PUSH2', '0xc4', 'JUMPI', 'PUSH1', '0x0', 'DUP1', 'REVERT']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH2', '0x4c', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0x5fd8c710', 'EQ', 'PUSH2', '0x51', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH2', '0x5d', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH2', '0x66', 'PUSH2', '0xb5', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'DUP1', 'PUSH1', '0x0', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'DUP2', 'MSTORE', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'DUP2', 'MSTORE', 'PUSH1', '0x20', 'ADD', 'PUSH1', '0x0', 'SHA3', 'SLOAD', 'SWAP1', 'POP', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'DUP2', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP8', 'GAS', 'CALL', 'SWAP3', 'POP', 'POP', 'POP', 'ISZERO', 'ISZERO', 'PUSH2', '0x12f', 'JUMPI', 'JUMPDEST', 'PUSH1', '0x0', 'DUP1', 'PUSH1', '0x0', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'DUP2', 'MSTORE', 'PUSH1', '0x20', 'ADD', 'SWAP1', 'DUP2', 'MSTORE', 'PUSH1', '0x20', 'ADD', 'PUSH1', '0x0', 'SHA3', 'DUP2', 'SWAP1', 'SSTORE', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'LT', 'ISZERO', 'ISZERO', 'PUSH1', '0x64', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'ISZERO', 'PUSH1', '0x64', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x1', 'ISZERO', 'ISZERO', 'PUSH1', '0x61', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'ISZERO', 'PUSH1', '0x61', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'LT', 'ISZERO', 'PUSH1', '0xaf', 'JUMPI', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'LT', 'ISZERO', 'PUSH1', '0xaf', 'JUMPI', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'PUSH1', '0xaf', 'JUMPI', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'PUSH1', '0xaf', 'JUMPI', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x1', 'ISZERO', 'PUSH1', '0xac', 'JUMPI', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x0', 'ISZERO', 'PUSH1', '0xac', 'JUMPI', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMPDEST', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'ISZERO', 'PUSH1', '0x64', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP']
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'LT', 'ISZERO', 'ISZERO', 'PUSH1', '0x64', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP'] # assert_operate_true
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x1', 'ISZERO', 'ISZERO', 'PUSH1', '0x61', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP'] #assert_true
# ['PUSH1', '0x80', 'PUSH1', '0x40', 'MSTORE', 'PUSH1', '0x4', 'CALLDATASIZE', 'LT', 'PUSH1', '0x3f', 'JUMPI', 'PUSH1', '0x0', 'CALLDATALOAD', 'PUSH29', '0x100000000000000000000000000000000000000000000000000000000', 'SWAP1', 'DIV', 'PUSH4', '0xffffffff', 'AND', 'DUP1', 'PUSH4', '0xa5333979', 'EQ', 'PUSH1', '0x44', 'JUMPI', 'JUMPDEST', 'CALLVALUE', 'DUP1', 'ISZERO', 'PUSH1', '0x4f', 'JUMPI', 'JUMPDEST', 'POP', 'PUSH1', '0x56', 'PUSH1', '0x58', 'JUMP', 'JUMPDEST', 'PUSH1', '0x2', 'PUSH1', '0x1', 'GT', 'ISZERO', 'ISZERO', 'PUSH1', '0x64', 'JUMPI', 'JUMPDEST', 'CALLER', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'PUSH2', '0x8fc', 'ADDRESS', 'PUSH20', '0xffffffffffffffffffffffffffffffffffffffff', 'AND', 'BALANCE', 'SWAP1', 'DUP2', 'ISZERO', 'MUL', 'SWAP1', 'PUSH1', '0x40', 'MLOAD', 'PUSH1', '0x0', 'PUSH1', '0x40', 'MLOAD', 'DUP1', 'DUP4', 'SUB', 'DUP2', 'DUP6', 'DUP9', 'DUP9', 'CALL', 'SWAP4', 'POP', 'POP', 'POP', 'POP', 'POP', 'JUMP'] #assert_opetate_false

opcodes_val_str = translate_bytecodes_to_opcodes(opcodes_list)

code = EVMAsm.assemble(opcodes_val_str)

data = constraints.new_array(index_bits=256, name="array",index_max=320,value_bits=8,taint=frozenset(),avoid_collisions=False)

class callbacks:
    initial_stack = []

    def will_execute_instruction(self, pc, instr):
        for i in range(len(evm.stack), instr.pops):
            e = constraints.new_bitvec(256, name=f"stack_{len(self.initial_stack)}")
            self.initial_stack.append(e)
            evm.stack.insert(0, e)

class DummyWorld(EVMWorld):
    def __init__(self, constraints):
        super().__init__(constraints)
        self.balances = constraints.new_array(index_bits=256, value_bits=256, name="balances")
        self.storage = constraints.new_array(index_bits=256, value_bits=256, name="storage")
        self.origin = constraints.new_bitvec(256, name="origin")
        self.price = constraints.new_bitvec(256, name="price")
        self.timestamp = constraints.new_bitvec(256, name="timestamp")
        self.coinbase = constraints.new_bitvec(256, name="coinbase")
        self.gaslimit = constraints.new_bitvec(256, name="gaslimit")
        self.difficulty = constraints.new_bitvec(256, name="difficulty")
        self.number = constraints.new_bitvec(256, name="number")

    def get_balance(self, address):
        return self.balances[address]

    def tx_origin(self):
        return self.origin

    def tx_gasprice(self):
        return self.price

    def block_coinbase(self):
        return self.coinbase

    def block_timestamp(self):
        return self.timestamp

    def block_number(self):
        return self.number

    def block_difficulty(self):
        return self.difficulty

    def block_gaslimit(self):
        return self.gaslimit

    def get_storage_data(self, address, offset):
        # This works on a single account address
        return self.storage[offset]

    def set_storage_data(self, address, offset, value):
        self.storage[offset] = value

    def log(self, address, topics, memlog):
        pass

    def send_funds(self, address, recipient, value):
        orig = self.balances[address] - value
        dest = self.balances[recipient] + value
        self.balances[address] = orig
        self.balances[recipient] = dest
    
    def send_funds(self, address, recipient, value):
        orig = self.balances[address] - value

caller = constraints.new_bitvec(256, name="caller")
value = constraints.new_bitvec(256, name="value")

# make the ethereum world state
# world = EVMWorld(constraints)

world = DummyWorld(constraints)
callbacks = callbacks()

# evm = world.current_vm
evm = EVM(constraints, 0x41424344454647484950, data, caller, value, code, world=world, gas=1000000)
print(evm.bytecode)
evm.subscribe("will_execute_instruction", callbacks.will_execute_instruction)

print("CODE:")
while not issymbolic(evm.pc):
    print(f"\t {evm.pc} {evm.instruction}")
    try:
        evm.execute()
    except:
        print('Error!')
    #     # print(type(e))
        break

# print translate_to_smtlib(arithmetic_simplifier(evm.stack[0]))
print(f"STORAGE = {translate_to_smtlib(world.storage)}")
print(f"MEM = {translate_to_smtlib(evm.memory)}")

for i in range(len(callbacks.initial_stack)):
    print(f"STACK[{i}] = {translate_to_smtlib(callbacks.initial_stack[i])}")
print("CONSTRAINTS:")
print(constraints)

smtsolver = Z3Solver()


print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
if smtsolver.check(constraints):    
    # print(smtsolver.get_value(constraints, evm.pc))
    print("msg.sender = {}".format(smtsolver.get_value(constraints, caller)))

smtsolver = YicesSolver()
print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
if smtsolver.check(constraints):    
    # print(smtsolver.get_value(constraints, evm.pc))
    print("msg.sender = {}".format(smtsolver.get_value(constraints, caller)))

smtsolver = CVC4Solver()
print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
if smtsolver.check(constraints):    
    # print(smtsolver.get_value(constraints, evm.pc))
    print("msg.sender = {}".format(smtsolver.get_value(constraints, caller)))
import os
import io
import json
import re
import pyevmasm
from pyevmasm import disassemble_hex
from pyevmasm import assemble_hex

# from solc import compile_source, compile_files, link_code
from solcx import compile_source, compile_files, link_code, get_installed_solc_versions, set_solc_version
from evm_cfg_builder.cfg import CFG
import sys

if __name__ == '__main__':

    file_name = sys.argv[1] # './reentrancy_eth.sol'
    compile_version = sys.argv[2] # 'v0.4.24'
    dot_dir = sys.argv[3] # 'reentrancy_eth/dot/reentrancy_eth.dot'

    # read the contract source code
    f = open(file_name)
    file_content = f.read()
    f.close()
    set_solc_version(compile_version)

    result = compile_source(file_content)

    # traverse the compiled contracts
    for k,v in result.items():
        print('contract', k)
        print("Runtime-bin bytecode:")
        print(v['bin-runtime'])

        cfg = CFG(v['bin-runtime'])
        print('block numbers', len(cfg.basic_blocks))
        
        print("outgoing_basic_blocks")
        for basic_block in cfg.basic_blocks:
            print('{} -> {}'.format(basic_block, sorted(basic_block.all_outgoing_basic_blocks, key=lambda x:x.start.pc)))

        print("incoming_basic_blocks")
        for basic_block in cfg.basic_blocks:
            print('{} -> {}'.format(basic_block, sorted(basic_block.all_incoming_basic_blocks, key=lambda x:x.start.pc)))

        dot_path = os.path.join(dot_dir, '{}.dot'.format(k))
        digraph_name = k
        node_connections = ""
        all_exit_node = -1
        output_exit = []
        output_exit_instructions = []
        jumpt_end = []
        end = []
        with open(dot_path, 'w') as f:
            f.write('digraph {} {{\n'.format(digraph_name))
            f.write('bgcolor=transparent rankdir=UD;\n')
            f.write('node [shape=box style=filled color=black fillcolor=white fontname=arial fontcolor=black];\n')
            for basic_block in cfg.basic_blocks:
                instructions = ['{}: {}'.format(ins.pc,
                                            str(ins)) for ins in basic_block.instructions]
                
                instructions = '\l'.join(instructions)
                instructions += '\l'
                if len(basic_block.all_incoming_basic_blocks) == 0:
                    if len(basic_block.all_outgoing_basic_blocks) == 0:
                        if str(basic_block.instructions[0]) == 'STOP':
                            all_exit_node = basic_block.instructions[0].pc
                            f.write('{} [label="{}: EXIT BLOCK\l" fillcolor=crimson ];'.format(all_exit_node, all_exit_node))
                        else:
                            f.write('{}[label="{}" fillcolor=lemonchiffon shape=Msquare ];\n'.format(basic_block.start.pc, instructions))
                            output_exit.append(basic_block.start.pc)
                            output_exit_instructions.append(instructions)
                            end.append(basic_block.start.pc)
                    else:
                        f.write('{}[label="{}" fillcolor=lemonchiffon shape=Msquare fillcolor=gold ];\n'.format(basic_block.start.pc, instructions))
                elif len(basic_block.all_outgoing_basic_blocks) == 0:
                    if len(basic_block.all_incoming_basic_blocks) != 0:
        #                 print(instructions)
                        if str(basic_block.instructions[-1]) == 'JUMP':
                            jumpt_end.append(basic_block.start.pc)
                            f.write('{}[label="{}" fillcolor=lemonchiffon ];\n'.format(basic_block.start.pc, instructions))
                        else:
                            end.append(basic_block.start.pc)
                            f.write('{}[label="{}" fillcolor=lemonchiffon shape=Msquare color=crimson ];\n'.format(basic_block.start.pc, instructions))
                else:
                    f.write('{}[label="{}" fillcolor=lemonchiffon ];\n'.format(basic_block.start.pc, instructions))

                for son in basic_block.all_outgoing_basic_blocks:
                    node_connections += '{} -> {};\n'.format(basic_block.start.pc, son.start.pc)
            
            for i in range(len(jumpt_end)):
                node_connections += '{} -> {};\n'.format(jumpt_end[i], output_exit[i])
            for end_val in end:
                node_connections += '{} -> {};\n'.format(end_val, all_exit_node)
                
            f.write(node_connections)
            f.write('}')

        print("{} has been saved successful!".format(dot_path))

        os.system('dot -T png {} -o {}'.format(dot_path, os.path.join(dot_dir, '{}.png'.format(k))))
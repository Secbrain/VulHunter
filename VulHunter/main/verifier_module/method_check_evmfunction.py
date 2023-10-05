from manticore.ethereum import (ManticoreEVM,
    ABI,
)
from manticore.core.smtlib import ConstraintSet, operators, PortfolioSolver, SolverType
from manticore.core.smtlib.visitors import to_constant
from manticore.ethereum.solidity import SolidityMetadata

contract_src="""
contract D {
    mapping(string => uint) storagelib;

    function set(string x, uint y) payable public {
        storagelib[x] = y;
    }

    function get(string x) payable public returns (uint) {
        return storagelib[x];
    }
}
"""

m = ManticoreEVM()

user_account = m.create_account(balance=10000000)
print("[+] Creating a user account", user_account)

print("[+] Creating a solidity_create_contract")
contract_account = m.solidity_create_contract(contract_src, owner=user_account, balance=0)
print(contract_account)

# x1 = m.make_symbolic_value() #符号值
# x1 = ABI.serialize("(uint)", 5) #数字
# print(x1)
# x1 = 5

md: SolidityMetadata = m.get_metadata(contract_account) #源码部署的时候可能才有

set_sig = ABI.function_selector("set(string,uint256)") #string,uint256 只能是uint256 这个是bytes
print("set_sig", set_sig)
set_sig_1 = md.get_hash("set(string,uint256)")
print("set_sig_1", set_sig_1)
set_func_sig = md.get_func_signature(set_sig)
print("set_func_sig", set_func_sig)
set_abi = md.get_abi(set_sig)
print("set_abi", set_abi)
func_arg_types = md.get_func_argument_types(set_sig)
print("func_arg_types", func_arg_types)
print("md.signatures", md.signatures) #map signature {b'i>\xc8^': 'get(string)', b'\x8aB\xeb\xe9': 'set(string,uint256)'}
function_signatures = md.function_signatures
print("md.function_signatures", function_signatures) #list dict_values(['get(string)', 'set(string,uint256)'])
set_return = md.get_func_return_types(set_sig)
print("set_return", set_return)
get_return = md.get_func_return_types(ABI.function_selector("get(string)"))
print("set_return", get_return)
print(md.has_non_default_constructor)
print(md.get_abi(md.fallback_function_selector)) #{'payable': False, 'stateMutability': 'nonpayable', 'type': 'fallback'}

sym_args = m.make_symbolic_arguments(func_arg_types)
print("sym_args", sym_args)
print("[+] Executing contract_account.set")
contract_account.set(sym_args[0], sym_args[1]) #这块之执行set的执行序列

# print("state nums ready set ", len(list(m.ready_states))) #1个元素
func_arg_types2 = md.get_func_argument_types(ABI.function_selector(list(function_signatures)[0]))
sym_args2 = m.make_symbolic_arguments(func_arg_types2)
print("sym_args2", sym_args2)
print("[+] Executing contract_account.get")
contract_account.get(sym_args2[0]) #光执行get

print("state nums ready get ", len(list(m.ready_states))) #两个 0 和 1

# def evm_trace_process(trace_list):
#     trace_list_new = []
#     contract_map = {}
#     contract_val = 0
#     for i in range(len(trace_list)):
#         if trace_list[i][0] not in contract_map:
#             contract_map[trace_list[i][0]] = contract_val
#             contract_val += 1
#         if not trace_list[i][2]:
#             trace_list_new.append((contract_map[trace_list[i][0]], trace_list[i][1]))
    
#     return trace_list_new

for state in m.ready_states:
    print("id ", state.id)
#     # print(state.context.get('evm.trace'))
    # print(state.context)

    world = state.platform

    tx = world.all_transactions[-1] #得到word中的交易
    retval_array = world.human_transactions[-1].return_data
    # retval = operators.CONCAT(256, *retval_array) #转换类型
    
    with state as tmp:
        tmp.constrain(retval_array == ABI.serialize("uint256", 5)) #添加约束 调用 或者直接5 retval转换一下
        inp = tmp.solve_one_n_batched(sym_args) #sym_args[0],sym_args[1]
        print("sym_args", inp)
        print("func_arg_types", func_arg_types)
        # print("sym_args", ABI.deserialize(func_arg_types, tuple(inp))[0])
        print("sym_args", ABI.deserialize('(string)', inp[0])[0]) #5不能反编译
        # print("sym_args", ABI.deserialize(func_arg_types, inp)[0])
        # print("sym_args", ABI.deserialize('string', inp[0])[0], ABI.deserialize('uint256', inp[1])[0])
        inp2 = tmp.solve_one_n_batched(sym_args2)
        print("sym_args2", inp2)
        print("func_arg_types2", func_arg_types2)
        print("sym_args2", ABI.deserialize(func_arg_types2, inp2[0])[0])
    # with state as sta:
        # print("meet the condition, value(x1,x2) = [{},{}]".format(x1,sta.solve_one(x2)))
        # print("meet the condition, value(x1,x2) = {}".format(sta.solve_one_n(x1, x2)))
    
#     # did_gen = m.generate_testcase(state, 'test' + str(state.id), name = "statecase" + str(state.id))
#     # print(did_gen.prefix)

#     # print(ABI.deserialize("(bool)", to_constant(retval_array))[0]) #用to_constant可以直接把静态值拿出来
#     print("meet the add condition, value = {}".format(ABI.deserialize("(bool)", state.solve_one(retval_array))[0])) 

#     print("---------------")
#     # break
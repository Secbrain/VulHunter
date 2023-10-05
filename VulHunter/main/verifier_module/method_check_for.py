from manticore.ethereum import (ManticoreEVM,
    ABI,
)
from manticore.core.smtlib import ConstraintSet, operators, PortfolioSolver, SolverType
from manticore.core.smtlib.visitors import to_constant

contract_src="""
contract C {
    uint n;

    function set(uint x) payable public {
        n = x;
    }

    function get(uint x) payable public returns (uint) {
        for(uint i=0;i<10;i++){
        	x += 1;
        }
        return x;
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

# print("[+] Executing contract_account.set")
# contract_account.set(x1) #这块之执行set的执行序列

# print("state nums ready set ", len(list(m.ready_states))) #1个元素

x2 = m.make_symbolic_value()

print("[+] Executing contract_account.get")
contract_account.get(x2) #光执行get

print("state nums ready get ", len(list(m.ready_states))) #两个 0 和 1

def evm_trace_process(trace_list):
    trace_list_new = []
    contract_map = {}
    contract_val = 0
    for i in range(len(trace_list)):
        if trace_list[i][0] not in contract_map:
            contract_map[trace_list[i][0]] = contract_val
            contract_val += 1
        if not trace_list[i][2]:
            trace_list_new.append((contract_map[trace_list[i][0]], trace_list[i][1]))
    
    return trace_list_new

for state in m.ready_states:
    print("id ", state.id)
    # print(state.context.get('evm.trace'))
    print(state.context)

    world = state.platform

    tx = world.all_transactions[-1] #得到word中的交易
    retval_array = world.human_transactions[-1].return_data

    with state as sta:
        sta.constrain(retval_array == ABI.serialize("uint256", 12))
        print("meet the condition, value(x2) = [{}]".format(sta.solve_one(x2)))
        # print("meet the condition, value(x1,x2) = {}".format(sta.solve_one_n(x1, x2)))
    
    # did_gen = m.generate_testcase(state, 'test' + str(state.id), name = "statecase" + str(state.id))
    # print(did_gen.prefix)

    # print(ABI.deserialize("(bool)", to_constant(retval_array))[0]) #用to_constant可以直接把静态值拿出来
    # print("meet the add condition, value = {}".format(ABI.deserialize("(bool)", state.solve_one(retval_array))[0])) 

    print("---------------")
    # break
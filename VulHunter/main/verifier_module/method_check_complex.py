from manticore.ethereum import (ManticoreEVM,
    ABI,
)
from manticore.core.smtlib import ConstraintSet, operators, PortfolioSolver, SolverType

contract_src="""
contract C {
    uint n;
    function C(uint x) {
        n = x;
    }
    function f(uint x) payable public returns (bool) {
        if (x == n) {
            return true;
        }
        else if (x == 3) {
            return true;
        }
        else{
            return false;
        }
    }
}
"""
m = ManticoreEVM()

user_account = m.create_account(balance=10000000)
print("[+] Creating a user account", user_account)

contract_account = m.solidity_create_contract(contract_src, owner=user_account, balance=0, args=[42])
print(contract_account)

value = m.make_symbolic_value()

contract_account.f(value)

# 读取ready_states 是否走到状态
# for state in m.ready_states:
#     print("can value be 42? {}".format(state.can_be_true(value == 42))) #给定输入可以退则是否能够执行出东西，进行验证，只不过路径约束是怎么来的
#     print("can value be 3? {}".format(state.can_be_true(value == 3)))
#     print("can value be 100? {}".format(state.can_be_true(value == 100)))
#     # print(state._constraints)
#     # print(state.get_value())

# print("----------")

# 读取all_states 是否走到状态
# for state in m.all_states:
#     print("can value be 42? {}".format(state.can_be_true(value == 42)))
#     print("can value be 3? {}".format(state.can_be_true(value == 3)))
#     print("can value be 100? {}".format(state.can_be_true(value == 100)))
#     # print(state._constraints)
#     # print(state.get_value())

# 读取all_states 反推value 三个状态对应三值具体
for state in m.all_states: #从mainticore中 _load index得到值
    # print(dir(state))
    with state as sta:
        print("meet the condition, value = {}".format(sta.solve_one(value)))
    # print(state.constraints)
    print(state.id)
    print(state.__str__)
    print(state._terminated_by)

    # print(state.context.get('evm.trace')[-1])

    print("transaction nums: ", len(state.platform.human_transactions))
    for i in range(len(state.platform.human_transactions)):
        retval_array = state.platform.human_transactions[i].return_data 
        # print(ABI.deserialize("(bool)", state.solve_one(retval_array)))
        # result_val = operators.CONCAT(256, *retval_array)
        
        print("transaction_{}: {}".format(i, ABI.deserialize("(bool)", state.solve_one(retval_array))[0]))

    # world = state.platform
    # tx = world.all_transactions[-1] #得到word中的交易
    # print(tx)

    # print(state.context) #evm的路径可以得到
    # print(state.input_symbols) #property直接用
    # print(state.is_feasible())

    # balance = state.platform.get_balance(int(user_account)) #得到balance的一个种子量 
    # print(state.solve_one(balance))

    # did_gen = m.generate_testcase(state, 'test', name = "statecase" + str(state.id)) #约束条件为None
    # print(did_gen.prefix)
    # m.generate_testcase(state, 'balance CAN be 0', only_if=balance == 0) #添加约束条件

    print("---------------")
    # break


# 读取所有返回值和state的id wasm
# for idx, val_list in enumerate(m.collect_returns()):
#     print("State", idx, "::", val_list)


# 对输入参数在上述基础上添加限制，再得到相应的值 wasm文件
# def arg_gen(state):
#     # Generate a symbolic argument to pass to the collatz function.
#     # Possible values: 4, 6, 8
#     # arg = state.new_symbolic_value(32, "collatz_arg")
#     state.constrain(value > 2)
#     state.constrain(value < 6)
#     # state.constrain(value % 2 == 0)
#     return [value]

# # Run the collatz function with the given argument generator.
# m.collatz(arg_gen)
# # Manually collect return values
# # Prints 2, 3, 8
# for idx, val_list in enumerate(m.collect_returns()):
#     print("State", idx, "::", val_list[0])
#     # break

# print("[+] Now the symbolic values")
# symbolic_data = m.make_symbolic_buffer(320)
# symbolic_value = m.make_symbolic_value(name="value")
# m.transaction(
#     caller=user_account, address=contract_account, value=symbolic_value, data=symbolic_data
# )

# print("[+] Resulting balances are:")
# for state in m.all_states:
#     balance = state.platform.get_balance(int(user_account))
#     print(state.solve_one(balance))

from manticore.ethereum import ManticoreEVM
from manticore.core.smtlib import ConstraintSet, operators, PortfolioSolver, SolverType

contract_src="""
contract C {
    uint n;
    function f(uint x, uint y) payable public returns (uint) {
        uint z = x + y;
        return z;
    }
}
"""
m = ManticoreEVM()

user_account = m.create_account(balance=10000000)
print("[+] Creating a user account", user_account)

contract_account = m.solidity_create_contract(contract_src, owner=user_account, balance=0)
# print(contract_account)

x = m.make_symbolic_value()
y = m.make_symbolic_value()

contract_account.f(x,y)

print("state length nums (all_states): ", len(list(m.all_states))) #三个状态 有两个else，仅返回一个

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
print("state length nums: ", len(list(m.ready_states))) #三个状态 有两个else，仅返回一个
for state in m.ready_states: #从mainticore中 _load index得到值 和all_states好像差不多
    # print(dir(state))
    print('id', state.id)
    
    print(state.constraints)

    print(state._terminated_by)

    # print(state.context.get('evm.trace'))

    with state as sta:
        world = sta.platform
        tx = world.all_transactions[-1] #得到word中的交易
        retval_array = world.human_transactions[-1].return_data
        result_val = operators.CONCAT(256, *retval_array)

        print("meet the condition, value = {}".format(sta.solve_one_n(x,y,result_val)))
        sta.constrain(operators.UGT(x, result_val)) #添加约束 调用
        # sta.constrain(x < (2^256)) #添加约束 调用
        # sta.constrain(y < (2^256)) #添加约束 调用
        print(sta.constraints)
        print("meet the add condition, value = {}".format(sta.solve_one_n(x,y,result_val))) 

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

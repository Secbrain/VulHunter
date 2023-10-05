from manticore.ethereum import ManticoreEVM
contract_src="""
contract Adder {
    function incremented(uint value) public returns (uint){
        if (value == 1)
            revert();
        return value + 1;
    }
}
"""
m = ManticoreEVM()

user_account = m.create_account(balance=10000000)
contract_account = m.solidity_create_contract(contract_src,
                                              owner=user_account,
                                              balance=0)
value = m.make_symbolic_value()

contract_account.incremented(value)

for state in m.ready_states:
    print("can value be 1? {}".format(state.can_be_true(value == 1)))
    print("can value be 200? {}".format(state.can_be_true(value == 200)))
    print(state._constraints)
    # print(state.get_value())

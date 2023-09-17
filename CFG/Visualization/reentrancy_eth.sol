pragma solidity ^0.4.24;

contract StandardToken {
	mapping (address => uint) private userBalances;

	address owner;

	modifier onlyOwner() {
		require(msg.sender == owner);
		_;
    }

	function transfer(address to, uint amount) {
		if (userBalances[msg.sender] >= amount) {
			userBalances[to] += amount;
			userBalances[msg.sender] -= amount;
		}
	}

	function modifierowner() public {
        // add administrators here
        owner = msg.sender;
    }

	function withdrawBalance() onlyOwner() public {
		uint amountToWithdraw = userBalances[msg.sender];
		if (!(msg.sender.call.value(amountToWithdraw)())) { throw; } // At this point, the caller's code is executed, and can call transfer()
		userBalances[msg.sender] = 0;
	}
}

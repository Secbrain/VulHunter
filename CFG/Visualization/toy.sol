contract Test{
    address owner;

    mapping (address => uint) balances;

    constructor(){
        balances[msg.sender] = 0;
	    owner = msg.sender;
    }

    function storage_coins(uint storage_value) public {
        balances[msg.sender] += storage_value;
    }

    function transfer_coins(uint transfer_value) public{ 
        if(balances[msg.sender] >= transfer_value){
            balances[msg.sender] -= transfer_value;
        }
    }
}

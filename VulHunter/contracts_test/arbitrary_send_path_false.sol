contract Test{

    address destination;
    address owner;

    mapping (address => uint) balances;

    constructor(){
        balances[msg.sender] = 0;
	    owner = msg.sender;
    }

    modifier isowner() {
        address aa = msg.sender;
        require(owner == aa);
        _;
    }

    function indirect() public{
        //assert(1<2);
        if(false){
            msg.sender.send(address(this).balance);
        }
        //msg.sender.send(address(this).balance);
    }
}

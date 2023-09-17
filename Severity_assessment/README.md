# The details of contract vulnerability severity assessment

Combined with CVSS2.0 (Common Vulnerability Scoring System), the severity of smart contract vulnerabilities (refer to contract bugs and code optimizations) can be rated as *High*, *Medium*, *Low*, *Informational* (*Info*), and *Optimization* (*Opt*) in terms of risk degrees and utilization difficulties. 

<img src=".\vulnerability_rating.png" style="text-align: center; zoom:50%;" />

## Risk degrees

The risk degree refers to the impact of vulnerability on the resources such as blockchain system and users. According to the three impact dimensions of confidentiality (C), integrity (I), and availability (A), this paper divides the risk degree into *High*, *Medium*, *Low*, *Informational* (*Info*) and *Optimization* (*Opt*). Specifically, *High* risk refers to the severe and almost irreversible harm, i.e., the vulnerability can seriously affect the CIA of smart contracts, and cause a lot of economic losses and data confusion to the contract business system. Including but not limited to: (i) the large assets being stolen or frozen; (ii) the core contract business cannot operate normally, such as denial of service; (iii) the core business logic of contracts is arbitrarily tampered with or bypassed, such as transfer, charging, and accounting; (iv) the fairness design of contracts is invalid, such as electronic voting, lottery, and auction.

*Medium* risk refers to a slight impact on the CIA of smart contracts, and may cause certain harm to the contract business system, such as little economic losses. Including but not limited to: (i) some assets being stolen or frozen; (ii) the non-core business logic of contracts is destroyed; (iii) the non-core business verification of contracts is bypassed; (iv) the contracts trigger error events or adopt non-standard interfaces, resulting in loss of the external system.

*Low* risk refers to a weak impact on the contract business system. Including but not limited to: (i) the stability of contract operation is affected, such as the abnormal increase in call failure rate and resource consumption; (ii) the contracts adopt substandard interfaces or their implementation, which affects the security and compatibility of interfaces; (iii) the contracts can trigger false events with few losses.

*Info* risk refers to hardly substantial harm to the contract business system and reminds contract developers that the contract code is prone to errors. Thus, the contract owners should develop the contracts following the specifications. Including but not limited to: (i) the sensitive function calls such as delegatecall; (ii) the precautions required by the security development specifications, such as variables and functions updated in version 0.5.0.

*Opt* risk refers to the improvement of the contracts, which can make the contract more efficient, readable, and less gas consumption. Including but not limited to: (i) remove useless operations and reduce operating overhead; (ii) optimize algorithm code to improve the running speed and security.

## Utilization difficulties

The utilization difficulty refers to the possibility of vulnerability occurrence. According to the three dimensions of attack cost (e.g., money, time, and technology), utilization condition (i.e., the difficulty of attack utilization), and trigger probability (e.g., vulnerabilities can only be triggered by a few users), this paper divides it into *exactly*, *probably* and *possibly*. Generally, *exactly* utilization requires an inferior cost, and the vulnerabilities can be stably triggered without a special threshold. Including but not limited to: (i) easy to invoke; (ii) need few costs; (iii) hold few assets.

*Probably* utilization requires a certain cost and utilization conditions, and the vulnerabilities are not easy to trigger. Including but not limited to: (i) pay a certain cost but less than attack proceeds; (ii) require attackers to achieve certain normal conditions, such as collusion with miners or outbound nodes; (iii) cooperate with known attacks in smart contracts, such as attacking other on-chain Oracle contracts.

*Possibly* utilization requires expensive costs and strict utilization conditions, and the vulnerabilities are more difficult to trigger. Including but not limited to: (i) pay the cost that more than the attack proceeds; (ii) require the attackers to meet low-frequency conditions, such as belonging to a few critical accounts, and constructing a difficult specific signature.

## Comparison with custom severity assessment methods

Currently, there are some evaluation schemes for contract vulnerability severity, such as Securify2.0 [1], DefectChecker [2], Slither [3], SmartCheck [4], etc. Among them, Securify2.0 divided the vulnerability severity into five levels (Critical, High, Medium, Low, and Info), and investigated the vulnerabilities from the SWC vulnerability database. SmartCheck divided it into three levels (High, Medium, and Low). Note that they are a preliminary attempt to assess contract vulnerabilities, which set a precedent. However, on the one hand, they lack the details of the assessment principles and the difference between the severity. On the other hand, from the developer's perspective, the severity should reflect the contract security comprehensively and directly, i.e., reporting the type of vulnerabilities, reminders, and optimization. This can help developers to optimize the contract more intuitively. 

To this end, DefectChecker considered the severity from three aspects of unwanted behavior (critical, major, trivial), attack vector (triggered by external, stolen ethers), usability (potential errors for callers, gas waste, mistakes on code reuse), and divided it into 5 levels of IP1-5. IP1 is the highest, and IP5 is the lowest. Contract defects with impact level 1-2 can lead to critical unwanted behaviors, like crashing or a contract being controlled by attackers. Contract defects with impact level 3 can lead to major unwanted behaviors, like lost ethers. Impact level 4-5 can lead to trivial problems, e.g., low readability, which would not affect the normal running of the contract. The mechanism details the evaluation basis and the distinction between the severity. Nonetheless, they still ignored the definition of optimization type. 

Recently, Slither has classified contract severity into High, Medium, Low, Informational, and Optimization from impact (High, Medium, Low, Informational, Optimization) and confidence (High, Medium, Low), which is consistent with our consideration. However, they lack the introduction of the evaluation details and judgment criteria. Overall, these mechanisms are formulated with their own considerations. Similar to them, we are working to find a suitable evaluation model that can describe the contract vulnerabilities precisely and obviously, thereby allowing contract developers or auditors to understand the contract security. Of course, this evaluation mechanism may not be perfect, yet it can give others some inspiration, enabling them to improve and propose a better method.

- [1] [Securify v2.0.](https://github.com/eth-sri/securify2) [EB/OL], SRI Lab, ETH Zurich, 20
- [2] [DefectChecker: Automated Smart Contract Defect Detection by Analyzing EVM Bytecode](https://ieeexplore.ieee.org/document/9337195), Jiachi Chen, Xin Xia, David Lo, John Grundy, Xiapu Luo, Ting Chena, Yujia Chen - IEEE TSE '20
- [3] [Slither: A Static Analysis Framework For Smart Contracts](https://arxiv.org/abs/1908.09878), Josselin Feist, Gustavo Grieco, Alex Groce - WETSEB '19
- [4] [SmartCheck: Static Analysis of Ethereum Smart Contracts](https://orbilu.uni.lu/bitstream/10993/35862/3/smartcheck-paper.pdf), Sergei Tikhomirov, Ekaterina Voskresenskaya, Ivan Ivanitskiy, Ramil Takhaviev, Evgeny Marchenko, Yaroslav Alexandrov - WETSEB '18

## Motivation examples

### The description of vulnerabilities

```solidity
contract PullPayment {
    mapping (address => uint) userBalances;
    function withdraw(){
    //Reenter the function
    if(!(msg.sender.call.value(userBalance[msg.sender])())){ throw; }
    	userBalance[msg.sender] = 0;
	}
}
contract Attack {
    PullPayment object;
        function attack() payable {
        object.withdraw(1 ether);
    }
    function() public payable {
    	object.withdraw(1 ether);
    } 
}
```
Reentrancy with Ether (*reentrancy-eth*). Reentrancy vulnerability is a classical problem, which leads to the asset loss of nearly $60 million in 2016 (Sergey and Hobor, 2017). This vulnerability refers to reentry with the following features: reentrant call, Ether sending, and reading the variables before writing. The above code describes an attack scenario. Bob first constructs a contract *Attack*, and then he performs the function ''withdraw()'' by invoking the attack(), which will trigger the fallback function. By this means, Bob implements multiple calls to withdraw(). Since the userBalance hadn't changed before the call in withdraw(), Bob obtained more than the amount he deposited into the contract. It should be noted that the Ether sent cannot be zero. Otherwise, it will cause false positives. Improvements to contracts: put userBalance[msg.sender]=0 before the call function. That is, the contracts should employ the check-effects-interactions pattern to avoid this vulnerability.

```solidity
contract Token {	
    address payable o; // owner
    mapping(address => uint) tokens;
    function withdraw() public {
        uint amount = tokens[msg.sender];
        address payable d = msg.sender;
        tokens[msg.sender] = 0;
        _withdraw(/*owner*/o ,d /*destination*/
        /*value*/, amount);
    }
}
```
Right-To-Left-Override (*rtlo*). This vulnerability can manipulate the logic of the contract by using a right-to-left-override character (U+202E). As shown in above code, the contract *Token* uses the right-to-left-override character when invoking _withdraw() function. As a result, the fee is incorrectly sent to msg.sender, and the token is sent to the owner. Improvements to contracts: Remove the special control characters (i.e., U+202E) in the contract.

```solidity
contract Locked{
    mapping(address => uint) tokens;
    function recharge() payable public{
    	tokens[msg.sender] += msg.value;
    }
}
```
Lock account assets (*locked-ether*). As shown in above code, since the recharge() has a tag ''payable'' (i.e., supports receiving remittance), everyone can transfer amounts to the contract *Locked* through this function. However, *Locked* doesn't provide any functions with withdrawal power. Thus, every Ether sent to *Locked* will be lost. Improvements to contracts: Remove the payable attribute or add a function "withdraw" in the contract.

```solidity
contract TxOrigin {
    address owner = msg.sender;
        modifier verify() {
        require(tx.origin == owner);
        _;
    }
}
```
Transaction origin address (*tx-origin*). As the underlying property of the transaction, the origin address may be manipulated by the attacker, so that it is inappropriate to be used for authentication. An attack scenario is depicted in above code. Bob is the owner of the contract *TxOrigin*. Bob calls Eve's contract. Eve's contract invokes TxOrigin and bypasses the tx.origin protection. Thus, the modifier "verify()" will lose its verification effect, making the contract abnormal. Improvements to contracts: Don't use tx.origin for authentication.

```solidity
contract C {
    function f() internal returns (uint b) {
        assembly {
        	b := shr(b, 8)
        }
    }
}
```
Wrong shift parameters (*shift-parameter-mixup*). The opcode shr(a,b) indicates that b is shifted right by a bits. However, the parameter errors for this opcode will get unexpected results. As shown in above code, the shift statement right-shifts the constant 8 by b bits due to the opposite position of the parameters. Then, the function "f()" returns an incorrect value, which may cause unexpected issues. Improvements to contracts: Swap the order of parameters to shr(8,b).

```solidity
contract ShadowedSC {
    uint now; //shadowing built-in symbols
    function get_random() private returns (uint) {
        require(owner == msg.sender); //invalidation
        return now + 259200; //unexpected result
    }
    function require(bool condition) public {
    	...;
    }
}
```
Shadowed built-in elements (*shadowing-builtin*). Solidity enables the shadowing of most elements in the contract, such as variables and functions. The shadowed elements may not be invoked as the user wishes and cause the wrong results. This vulnerability indicates that the names of the elements (e.g., state variables and custom functions) conflict with the built-in symbols (e.g., ''assert'', ''now'', ''sha3''). Moreover, ''try'', ''case'' and other reserved keywords are not recommended when defining the element name. The above code shows an example of this vulnerability. Since the state variable and the time variable ''now'' have the same name, the current time will not be obtained when function ''get_random()'' is invoked. Moreover, the built-in symbol ''require'' is hidden by the function defined by the contract, which may cause the invalid authentication in get_random(). In short, these shadowed built-in elements lead to unexpected results. Improvements to contracts:} Modify or delete declared elements such as ''now'' and ''require(bool)''.

```solidity
function transfer(bytes _signature,address _to,uint256 _value,uint256 _gas,uint256 _nonce) public returns (bool){
    bytes32 txid = keccak256(abi.encodePacked(getTransferHash(_to, _value, _gas, _nonce), _signature));
    require(!signatureUsed[txid]);
    address from_address = recoverTransferPreSigned(_signature, _to, _value, _gasPrice, _nonce);
    balances[from_address] -= _value;
    balances[_to] += _value;
    signatureUsed[txid] = true;
}
```
Non-compliant signature (*signature-malleability*). The use of signatures should follow the norms. Otherwise, it will cause unexpected impacts. The above code shows an incorrect signature example. That is, keccak256 contains the existent signature ''_signature''. The attackers can modify the elements r\s\v in _signature to construct a new valid signature, which can get a valid address from the function ''recoverTransferPreSigned''. As _signature changes, the verification (line 3) will be passed, thereby allowing the attackers to get additional balance. Improvements to contracts: Remove _signature in keccak256.

```solidity
contract Timestamp{
    uint time_now = 1577808000;
    address private receiver;
    function frangibility() public {
        require(block.timestamp > time_now);
        uint 1_ther = 10000000000000000000;
        receiver = msg.sender;
        require(receiver.call.value(1_ther).gas(7777)(""));
    }
}
```
Timestamp dependency (*timestamp*). The object ''block'' contains many attributes (e.g., timestamp and block number), which can be manipulated by miners and nodes. As shown in above code, attackers can control the block.timestamp to pass the verification (line 5) by conspiring with miners or nodes. Also, the contract owner confuses users with complex numbers (too-many-digits) and guided names, where the variable 1_ether is actually 10 ether. Furthermore, the function ''frangibility()'' uses this variable and the calls with poor safety (low-level-calls), which may cause security issues such as *reentrancy-eth*. In addition, although the visibility of the state variable ''receiver'' is declared private, miners can still be viewed in advance through the transaction (*private-not-hidedata*). Improvements to contracts: Use the scientific notation to modify the statement to 10_ether=$10^{19}$. Moreover, the calls with low-level security and block attributes such as timestamp should be avoided. Also, it is recommended to use private data in the ciphertext.

<img src=".\code_no_effect.png" style="text-align: center; zoom:25%;" />

Useless code (*code-no-effects*). The execution of each operation in the contract will consume the specified gas. However, as shown in above figure, useless_variable is clarified and without any use, which makes the gas of the function ''bad()" wasted. Moreover, state variables consume more gas than local variables, so frequent manipulation of state_variable in the loop consumes additional gas (costly-operations-loop). Improvements to contracts: As shown in above figure, the useless variable was removed. Also, the local variable ''tmp_variable'' was updated and then assigned to the state variable.

### The judgment basis

**Vulnerability** | **Description** | **Risk** | **Utilization** | **Severity**
--- | --- | --- | --- | ---
rtlo                   | Right-To-Left-Override control character is used.                                | High          | exactly                | High   
reentrancy-eth         | Ether stolen illegally by re-entering functions such as calls.                      | High          | probably               | High     
locked-ether           | The contract has a payment function but without withdrawal capacity.                  | Medium        | exactly                | Medium   
tx-origin              | Verification based on tx.origin may be bypassed.                                 | Medium        | probably               | Medium   
shift-parameter-mixup  | The parameter errors of shift operation cause the opposite results.              | Medium        | possibly            | Low      
shadowing-builtin      | The names of elements (\eg state variables) conflict with the built-in symbols. | Low           | exactly                | Low      
timestamp              | Miners or nodes can manipulate block.timestamp to achieve their purpose.         | Low           | probably               | Low      
signature-malleability | The signature to be verified contains an existing signature.                     | Low           | possibly            | Low      
low-level-calls        | Low-level functions such as call and delegatecall are used in the contract.      | Info          | exactly                | Info     
too-many-digits        | Complex number symbols will confuse contract users.                              | Info          | probably               | Info     
private-not-hidedata   | Private visibility doesn't guarantee data confidentiality.                       | Info          | possibly               | Info     
code-no-effects        | The useless code will increase gas consumption during running.                   | Opt           | exactly                | Opt      
costly-operations-loop | Redundant operations in the loop will waste resources such as gas.               | Opt           | probably               | Opt 

The above table details the assessment results, indicating that these vulnerabilities have different risk degrees and utilization difficulties (i.e., there are 13 combinations). Thus, they are given various severity. Specifically, the *rtlo* vulnerability can destroy the core business logic and the fair design (*High* risk); it is well-characterized and can be performed deterministically (*exactly* utilization) after the contract deployment.

The *reentrancy-eth* vulnerability can cause massive assets overspent or stolen (*High*); some conditions are required to trigger this vulnerability (*probably*). For instance, completing the attack requires auxiliary contracts.

The *locked-ether* vulnerability can make little assets be lost or frozen (*Medium*); it can be triggered stably after contract deployment (*exactly*).

The *tx-origin* vulnerability can cause the non-core business verification to be bypassed without direct economic losses (*Medium*); it requires a combination of ancillary contracts to obtain transactions from the verified party (*probably*).

Similar to *locked-ether*, the *shift-parameter-mixup* vulnerability can destroy the non-core business logic (*Medium*). Nonetheless, it requires that users are unclear about the parameters of shr(), thus giving it an inferior trigger probability (*possibly*).

The *shadowing-builtin* vulnerability can introduce security risks such as unexpected results and verification failure, so as to affect the stability of the contract operation (*Low*); it is triggered by executing the vulnerability code after the contract deployment (*exactly*).

The *timestamp* vulnerability can introduce security risks such as verification failure and random number utilization (*Low*); it requires attackers to collude with miners or block nodes (*probably*).

The *signature-malleability* vulnerability can cause a signature replay attack that affects the stability of contract operation (*Low*). However, the eligible signature is difficult to be constructed and exploit (*possibly*).

The *low-level-calls* vulnerability is prone to cause contract errors, so it is necessary to focus on the use of these functions (*Info*); it can be invoked by executing the vulnerable code (*exactly*).

The *too-many-digits* vulnerability can make users challenging to read the contracts, and inaccurate variable names will mislead them (*Info*); it requires users to misunderstand the semantic information of the variable, thus giving it a weak trigger probability (*probably*).

The *private-not-hidedata* vulnerability can result in the disclosure of private data, thus reminding us that private content should be stored in ciphertext (*Info*); it requires attackers to meet the specific identity restrictions and combine with miners to get information (*possibly*).

The *code-no-effects* vulnerability will cause useless gas consumption, which can be further optimized for user convenience (*Opt*); it can be improved after the contract development (*exactly*).

Similar to *code-no-effects*, the *costly-operations-loop* vulnerability can be further optimized to reduce the consumed gas (*Opt*); it depends on the number of loops, thus giving it a certain trigger probability (*probably*).
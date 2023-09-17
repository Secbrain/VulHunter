# Example_results

The folder "Example_results" describes the detection result of each method for examples involved in the paper.

## dataset3_and_5_examples.xlsx

The file "dataset3_and_5_examples.xlsx" describes the detection situation of each method. Among them, TP means the method that can truly detect the vulnerability, FN means the method that missed the vulnerability, and FP means the method that misreported the benign contract. It includes the Ethereum contracts in Dataset_3 and the vulnerability event contracts in Dataset_5.

## The detection details of additional examples

In addition to the examples of *reentrancy-eth* and *tod* vulnerabilities explained in the paper, this page describes the detection details of other contracts listed in Table 11. 

## Detection of *integer-overflow* vulnerabilities. 

```solidity
mapping (address => uint256) invested;
mapping (address => uint256) atBlock;
function () external payable {
	uint256 amount = invested[msg.sender] * 4 / 100 * (block.number - atBlock[msg.sender]) / 5900;
	msg.sender.transfer(amount);
	...
}
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The contract <i>WinStar</i> with <i> integer-overflow</i>.</div>
</div>

The contract *WinStar* (1.60E+15Wei) involves 25 transactions and has an *integer-overflow* vulnerability triggered by atBlock[msg.sender]=1 and invested[msg.sender]=115792089237316195423570985008687907853269984665640564039457584007913129639935 in line 4 of above listing (corresponding to line 43 in the contract). The vulnerability will make users extract the incorrect value of "amount''. It can be detected by VulHunter, but not by arts like Mythril, SMARTIAN and TMP. Similar contracts include *ETHMaximalist* (line 171), *AceTokensden* (line 255), *Eighterbank* (line 585), and *AceDapp* (line 242). For them, VulHunter can accurately identify the vulnerable contracts, while methods such as Slither and ContractWard are helpless. As we all know, Oyente can detect the *integer-overflow* vulnerability. However, against the *DharmaTradeReserve* contract (balance 9.63E+19Wei), it reported that there is an *integer-overflow* vulnerability in line 57 with the code string(returnData) (as shown in below listing). This is a false positive for Oyente. As expected, VulHunter detects that this code is secure. Similar contracts misreported by Oyente and TMP also include *Trader*, *DharmaTradeReserveStaging*, and *DeFi*. 

```solidity
function _implementation() ... returns (address ...) {
	(bool ok, bytes memory returnData) = _UPGRADE_BEACON.staticcall("");
	require(ok, string(returnData));
	...
}
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The contract <i>DharmaTradeReserve</i> with <i> integer-overflow</i>.</div>

## Detection of *locked-ether* vulnerabilities. 

VulHunter identified a *locked-ether* vulnerability in the *SavingAccountProxy* contract (2.15E+18Wei). In this contract, the ''initialize()'' function (line 339) can receive Ether value. However, the contract does not hold a withdrawal function, so users cannot retrieve the Ether exploited to invoke the contract. The methods such as Slither, Securify, and DefectChecker cannot identify this vulnerability. Similar vulnerable contracts include *Proxy* and *KyberFeeBurner*. 

## Detection of *uninitialized-state* vulnerabilities. 

```solidity
mapping(address => uint256) balances;
uint256 totalSupply_;
function totalSupply() public ... {return totalSupply_;}
function transfer(address _to, uint256 _value) public returns (bool) {
	...
	balances[msg.sender] = balances[msg.sender].sub(_value);
	...
}
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The contract <i>Token</i> with <i>uninitialized-state</i>.</div>
</div>

As shown in above listing, in the ''totalSupply()'' function of the *Token* contract (1.10E+16Wei), the ''totalSupply\_'' variable is invoked directly without initializing, which violates the development specification. Also, the contract does not provide a function to initialize the mapping variable ''balances'', so that the ''transfer()'' function fails while invoking it. It also applies to the ''balances'' variable and ''balanceOf'' function in the *ERC223I* contract. Therefore, we should develop contracts in compliance with the development specifications. For these vulnerabilities, VulHunter can provide warnings, which is impossible with methods such as Slither and ContractWard. 

## Detection of *unused-state* optimizations. 

In addition to discovering vulnerabilities, VulHunter can also assist with code optimization. It detects that the state variables such as ''week'' (line 2,979 of the *Oracle* contract) are declared but not used, which wastes extra gas. Also, the ''GSNRecipient'' contract can be optimized. However, methods like SmartCheck cannot support this detection. Although Slither can identify these defects, it located the state variables such as ''\_SHOULD\_OVERRIDE'' that have been used in the *Timelocker* contracts, which are false positives. 
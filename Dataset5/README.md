# File and Folder introduction of Dataset5

## Contracts
The folder "Contracts" includes 29 smart contract source codes of well-known vulnerability events and their solc versions in Dataset_5.

## Detection_result

The folder "Detection_result" describes the detection result of each method in detail.

## Extracted_instances

The folder "Extracted_instances" shows the instances extracted by VulHunter.

## Methods_detection

The folder "Methods_detection" includes the detection results of each method. Among them, the file "dataset5_result_Mythril.rar" shows the detection result files, the file "Mythril_audit_files_results_dataset5.json" shows the overall situation of the file audit, and the file "Mythril_files_holeloops_dataset5.json" shows the summary results of the method audit vulnerabilities.

# The detection details of additional incidents in Dataset5

In addition to the example of *integer-overflow* vulnerability incident explained in the paper, this page describes the detection details of other incidents listed in Table 12. 

## The *DOS* vulnerability indicident for the *KotET* contract.

```solidity
address public currentLeader;
uint256 public highestBid;
function bid() public payable {
	require(msg.value > highestBid);
	require(currentLeader.send(highestBid));//Dos code
	currentLeader =msg.sender;
	highestBid = msg.value;
}
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The <i>KotET</i> contract library for <i>DOS</i> incident.</div>
</div>

*KotET* designed a game ''throne race'', i.e., the player with the largest amount of competition will win the throne. However, in February 2016, the players could not obtain the throne, no matter how much ETH they sent to the contract. Specifically, the vulnerable contract is shown in above listing. The attacker can first construct an attacking contract that contains a fallback function executing the ''revert()'' operation, and then invoke the ''bid()'' function via this contract to become the king. In this way, the ''send()'' operation (line 12) will trigger the fallback function and always return false, intercepting the execution of ''bid()''. As a result, VulHunter benefits from its perfect feature learning ability to discover this vulnerability, yet the arts like Slither, SmartCheck, and Oyente missed it. 

## The *parity-multisig-bug* vulnerability indicident for Parity wallet. 

```solidity
function initWallet(...) {    
	initMultiowned(...);
}
function initMultiowned(...) {
	m_numOwners = _owners.length + 1;
	m_owners[1] = uint(msg.sender);
	m_ownerIndex[uint(msg.sender)] = 1;    
}
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The <i>Parity</i> contract library for <i>parity\_multisig\_bug</i> incident.</div>
</div>

In July 2017, a permission vulnerability in the Party contract library was exploited by attackers, resulting in the theft of over \$30 million ETH. Cause of the vulnerability: as shown in above listing, since the visibility of the ''initWallet'' function is declared as public, attackers can change the owner of the wallet by invoking the initWallet. VulHunter has identified this vulnerable contract through an automated feature learning process on this vulnerability dataset. In contrast, methods like Slither and Oyente without the ability to detect this vulnerability.

## The second Parity wallet vulnerability incident. 

```solidity
function initWallet(...) only {
	initMultiowned(...);
}
function initMultiowned(...) only { ...	}
modifier only { if (m_numOwners > 0) throw; _; }
```
<div align=left>
	<div style="color:orange; border-bottom: 1px solid #d9d9d7;
	display: inline-block;
	color: #999;
	padding: 2px;">The <i>Parity</i> contract library for <i>parity\_multisig\_bug\_2</i> incident.</div>
</div>

Towards repairing the above vulnerability, Parity supplemented the ''only'' modifier (line 12 in above listing) to restrict the ''initMultiowned()'' function to be invoked only once. However, in November 2017, the revised contract caused about \$152 million ETH to be frozen due to a permission vulnerability. Cause of the vulnerability: as shown in above listing, an attacker can invoke the ''initMultiowned()'' through the attacking contract. Since the variables such as ''m\_numOwners'' are operated from the attacking contract (i.e., m\_numOwners=0), the ''only'' modifier will be passed, and the owner of the library contract will be updated to the attackers. Fortunately, VulHunter was able to identify this vulnerability, demonstrating its superior identification ability for vulnerabilities. 

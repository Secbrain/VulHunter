# File and Folder introduction of Verification

## Contracts
The folder "Contracts" includes 42 (18+24) smart contract source codes of contracts used in path ablation and verification experiments. Also, this folder provides the Solc versions of these contracts.

## Methods_detection

The folder "Methods_detection" includes the detection results of each method. Among them, the file "Verification_result_Mythril.rar" shows the detection result files, the file "Mythril_audit_files_results_verification.json" shows the overall situation of the contract audit, and the file "Mythril_files_holeloops_verification.json" shows the summary results of the vulnerability detection.

## Extracted_instances

The folder "Extracted_instances" shows the instances extracted by VulHunter.

## Code_examples

The folder "Code_examples" includes some script examples for validating the feasibility of instances. They are developed based on a tool called Manticore.

## Ablation_experiments

The folder "Ablation_experiments" illustrates the detection results of methods in the path ablation experiment, which aim to explore the location of key instances in cooperation vulnerabilities. 

## Infeasible_paths

The folder "Infeasible_paths" details the detection results of methods in the verification experiment for infeasible paths. The contract examples in this experiment demonstrate the practicality of the constraint-solving module in VulHunter.

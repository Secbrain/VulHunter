# File and Folder introduction of Dataset1

## Contracts

The folder "Contracts" includes 38,600 smart contract source codes and their solc versions in Dataset_1.

## Detection_result

The folder "Detection_result" describes the detection result of each method in detail.

## Extracted_instances

The folder "Extracted_instances" shows the instances extracted by VulHunter.

## Labels

The folder "Labels" shows the labels of contract vulnerabilities in Dataset_1. The labels were manually checked and supplemented by the verification results of muti-methods such as Slither and Oyente. They are publicly available to enable cross-checks from other researchers and further guarantee their accuracy. In addition, the file "dataset_1_vul_two_one_names_labels.json" describes the contracts' names of the training dataset and test dataset used in the Benign:Malicious=2:1 experiment. And the file "dataset_1_vul_five_one_names_labels.json" shows that of the Benign:Malicious=5:1 experiment.

## Methods_detection

The folder "Methods_detection" includes the detection results of each method. Among them, the file "dataset1_result_2000_Mythril" shows the 2,000 detection result files, the file "Mythril_audit_files_results_dataset1.json" shows the overall situation of the file audit, and the file "Mythril_files_holeloops_dataset1.json" shows the summary results of the method audit vulnerabilities. In addition, the models in VulHunter are trained on Benign:Malicious=5:1 contracts in Dataset_1, and models of 2:1 are shown in the code folder. Meanwhile, the file folders "other_results_2vs1" and "other_results_5vs1" in ML-based methods (i.e., ContractWard, TMP, and VulHunter) include the detection results of other executions.
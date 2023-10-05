# VulHunter, an detection method based on multi-instance learning and machine learning for Ethereum smart contracts

<img src="./logo.png" alt="Logo" align="left" width="350"/>

VulHunter is a method that can effectively detect bytecode/opcode paths that trigger vulnerabilities without manual pre-defined patterns. It extracts the instances by traversing the Control Flow Graph built from contract opcodes. Based on the hybrid  Bag-instance/self-model attention and multi-instance learning (MIL) mechanisms, it reasons the instance labels and designs an optional classifier to automatically capture the subtle features of both normal and defective contracts, thereby identifying the vulnerable instances. Then, it combines the symbolic execution to construct and solve symbolic constraints to validate their feasibility. It is written in Python 3 with15K lines of code. Notably, it can detect contract vulnerabilities more accurately, efficiently, and robustly than the SOTA methods. Also, it can focus on specific metrics such as precision and recall by employing different baseline models and hyperparameters to meet the various user requirements, e.g., vulnerability discovery and misreport mitigation. More importantly, compared with the previous ML-based arts, it can not only provide classification results, defective contract source code statements, key opcode fragments, and vulnerable execution paths, but also eliminate misreports and facilitate more operations such as vulnerability repair and attack simulation during the contract lifecycle. 

<img src="./overviewer.png" alt="Logo" width="1100"/>

- [Bugs and Optimizationimizations Detection](#bugs-and-Optimizationimizations-detection)
- [How to Install](#how-to-install)
- [Publications](#publications)

## Bugs and Optimizationimizations Detection

### The model trainning of VulHunter 

The examples of Bi $^2$-LSTM detection models are stored in the dir of "models", which trainned based on contracts of the Dataset\_1.  Users can train the new models based on the pre-collected contract datasets through the following steps. First, extract the contract instances via the following command. Note that using GPU can improve the speed of the training and testing process. 

```bash
python3 main/main.py --train-contracts contract --train-solcversions solcversions.json --instance-dir input
```
Among it, the "contract" is the root dir of contract datasets (e.g., https://github.com/Secbrain/VulHunter/tree/main/Dataset1/Contracts/dataset1_sourcecodes), "solcversions.json" (e.g., https://github.com/Secbrain/VulHunter/tree/main/Dataset1/Contracts/solc_versions.json) includes their solc versions, and "input" is the output dir path of the extracted contract instances. The result outputs will include two files when the extraction is complete. That is, contract_bytecodes_list10.json includes the contract instances and contract_falsecompile_list.json shows the wrong extracted contracts. 

Note that the instance building module currently is an extensible framework, and it can support the evm_cfg_builder (original) and Ethersolve (run main/main_ethersolve.py, which uses the main/instance_generates/bytecodes_construction_Ethersolve.py, as shown in the following command) to build the CFG of contracts and traverse the instances.

```bash
python3 main/main_ethersolve.py --tmp-dir tmp_figs1 --cfg-dir ./main/instance_generates --train-contracts contract --train-solcversions solcversions.json --instance-dir input
```

Besides, some parameters can be considered such as --instance-len 10 (default number) to extract the customized contract instances. Also, the function codes of extraction methods such as the shortest and longest can be viewed in the main/bytecodes_construction.py. In total, the specific parameters can be viwed by executing the command python3 main/main.py --help.

Then, train the detection models based on the extracted instances of contracts and their labels.

```bash
python3 main/main.py --train-labels input/dataset_1_vul_two_one_names_labels.json --contract-instances contract_bytecode_list10.json --model-dir model_train
```
Among it, the file "contract_bytecode_list10.json" is an example of contract instances (https://github.com/Secbrain/VulHunter/tree/main/Dataset1/Extracted_instances/dataset1_instances.rar), which can be obtained by executing the above operations. Also, the file "dataset_1_vul_two_one_names_labels.json" includes the vulnerability labels of contracts, and "model_train" is the output dir of trained detection models. Similarly, some parameters can be considered such as --detectors reentrancy-eth,controlled-array-length  (default: all of detected vulnerabilities), --batchsize 512 (default number), and --epoch 50 (default number) to extract the customized contract instances. All parameters can be viewed by executing the command python3 main/main.py --help.

Besides, the model detection module currently is also an extensible framework, and it can support DNN models (e.g., the original Bi $^2$-LSTM, CNN, and Bi $^2$-GRU, which runs main/main_bi2gru.py and uses the main/model_types/result_predict_bigruatt.py, as shown in the following command) and traditional ML models (e.g., Random Forest, Decision Tree, XGB, KNN, and SVM, as stored in the dir main/model_types) to build the CFG of contracts and traverse the instances. Note that, the models Bi $^2$-LSTM and Bi $^2$-GRU can output the input weights, which can be used to locate the defective contract source codes. In the future, visualization tools such as Captum may help other models obtain the importance distribution of inputs.

```bash
python3 main/main_bi2gru.py --train-labels input/dataset_1_vul_two_one_names_labels.json --contract-instances contract_bytecode_list10.json --model-dir model_train
```

### The model detecting of VulHunter 

Run VulHunter with the Bi $^2$-LSTM model on a single solidity contract file:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10
```

Run VulHunter with the Bi $^2$-LSTM model on a single bytecode contract file:

```bash
python3 main/main.py --contract contracts_test/4500858670586072846.bin --filetype bytecode --model-dir models --instance-len 10 --ifmap nomap
```

Run VulHunter with the Bi $^2$-LSTM model on a single opcode contract file:

```bash
python3 main/main.py --contract contracts_test/arbitrary_send.evm --filetype opcode --model-dir models --instance-len 10 --ifmap nomap
```

Run VulHunter with the Bi $^2$-LSTM model on a single opcode contract file for detecting the specific vulnerabilities:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10 --detectors reentrancy-eth,controlled-array-length
```

Run VulHunter with the Bi $^2$-LSTM model on a single solidity contract file without defective source code statements:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10 --ifmap nomap
```

Run VulHunter with the Bi $^2$-LSTM model and output the result report without defective source code statements:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --filetype solidity --model-dir models --instance-len 10 --report reentrancy_eth_0424_nomap.pdf --ifmap nomap
```

Run VulHunter and output the result report with defective source code statements (default setting: --ifmap map):

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --filetype solidity --model-dir models --tmp-dir tmp_figs --instance-len 10 --report reentrancy_eth_0424_map.pdf
```

Run VulHunter with the Bi $^2$-LSTM model and output the main result report without defective source code statements:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --filetype solidity --model-dir models --instance-len 10 --report-main reentrancy_eth_0424_nomap_main.pdf --ifmap nomap
```

Run VulHunter and output the main result report with defective source code statements:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --filetype solidity --model-dir models --instance-len 10 --tmp-dir tmp_figs --report-main reentrancy_eth_0424_map_main.pdf
```

Run VulHunter and output the both result reports with defective source code statements:

```bash
python3 main/main.py --contract contracts_test/reentrancy_eth_0424.sol --filetype solidity --model-dir models --instance-len 10 --tmp-dir tmp_figs --report reentrancy_eth_0424_map.pdf --report-main reentrancy_eth_0424_map_main.pdf
```

Run VulHunter with the Bi $^2$-LSTM model and automatically verify the feasibility of vulnerable instances:

```bash
python3 main/main.py --contract contracts_test/arbitrary_send_path_false.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10 --verify
```
Among it, the optional verification module is used to build the feasibility of vulnerable paths and employ the solvers (e.g., Z3, Yices, CVC4) to validate their feasibility. Note that the default solver is Z3, and users can execute the following command to select the specific solvers.

```bash
python3 main/main.py --contract contracts_test/arbitrary_send_path_false.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10 --verify --solver Z3,Yices,CVC4
```

For more verification functions, users can view the scripts in the folder main/verifier_module, which can implement source code-level verification and solve the parameters that trigger the vulnerabilities.

In addition, similar to the training process, VulHunter can employ the other instance extractors and detection models to complete the contract audition. For example, the Ethersolve of instance extractor can be used to detecte contracts by executing the following command.

```bash
python3 main/main_ethersolve.py --contract contracts_test/reentrancy_eth_0424.sol --solc-version 0.4.24 --filetype solidity --model-dir models --tmp-dir tmp_figs1 --instance-len 10
```

The Bi $^2$-GRU of detection model can be used to detecte contracts by executing the following command.

```bash
python3 main/main_bi2gru.py --contract contracts_test/reentrancy_eth_0424.sol --solc-version 0.4.24 --filetype solidity --model-dir models --instance-len 10
```

Notably, for them, the other parameters detailed above can also be used to deliver the custom detection. Before using them to detect contracts, VulHunter should employ them to train the corresponding detection models based on the above training process.

### Detectors

The supported detectors can be viewed by executing the following command.

```bash
python3 main/main.py --list-detectors
```

The details of detectors are illustrated in the following table.

Num | Detector | What it Detects | Impact | Confidence | Severity
--- | --- | --- | --- | --- | ---
1 | `reentrancy-eth` | Re-entry vulnerabilities (Ethereum theft) | High | probably | High
2 | `controlled-array-length` | Length is allocated directly | High | probably | High
3 | `suicidal` | Check if anyone can break the contract | High | exactly | High
4 | `controlled-delegatecall` | The delegate address out of control | High | probably | High
5 | `arbitrary-send` | Check if Ether can be sent to any address | High | probably | High
6 | `tod` | Transaction sequence dependence for receivers/ethers | High | probably | High
7 | `uninitialized-state` | Check for uninitialized state variables | High | exactly | High
8 | `parity-multisig-bug` | Check for multi-signature vulnerabilities | High | probably | High
9 | `incorrect-equality` | Check the strict equality of danger | Medium | exactly | Medium
10 | `integer-overflow` | Check for integer overflow/overflow | Medium | probably | Medium
11 | `unchecked-lowlevel` | Check for uncensored low-level calls | Medium | probably | Medium
12 | `tx-origin` | Check the dangerous use of tx.origin | Medium | probably | Medium
13 | `locked-ether` | Whether the contract ether is locked | Medium | exactly | Medium
14 | `unchecked-send` | Check unreviewed send | Medium | probably | Medium
15 | `boolean-cst` | Check for misuse of Boolean constants | Medium | probably | Medium
16 | `erc721-interface` | Check the wrong ERC721 interface | Medium | exactly | Medium
17 | `erc20-interface` | Check for wrong ERC20 interface | Medium | exactly | Medium
18 | `costly-loop` | Check for too expensive loops | Medium | possibly | Low
19 | `timestamp` | The dangerous use of block.timestamp | Low | probably | Low
20 | `block-other-parameters` | Hazardous use variables (block.number etc.) | Low | probably | Low
21 | `calls-loop` | Check the external call in the loop | Low | probably | Low
22 | `low-level-calls` | Check low-level calls | Info | exactly | Info
23 | `erc20-indexed` | ERC20 event parameter is missing indexed | Info | exactly | Info
24 | `erc20-throw` | ERC20 throws an exception | Info | exactly | Info
25 | `hardcoded` | Check the legitimacy of the address | Info | probably | Info
26 | `array-instead-bytes` | The byte array can be replaced with bytes | Opt | exactly | Opt
27 | `unused-state` | Check unused state variables | Opt | exactly | Opt
28 | `costly-operations-loop` | Expensive operations in the loop | Opt | probably | Opt
29 | `send-transfer` | Check Transfe to replace Send | Opt | exactly | Opt
30 | `boolean-equal` | Check comparison with boolean constant | Opt | exactly | Opt
31 | `external-function` | Public functions can be declared as external | Opt | exactly | Opt

## How to install

VulHunter requires Python 3.7+, PyTorch 1.10.1+, [solc](https://github.com/ethereum/solidity/) and pyevmasm 0.2.3+. Among them, the py-solc-x 0.10.1+ is used to compile source code into bytecode, utilize pyevmasm to disassemble bytecode into opcode, and then use evm\_cfg\_builder 0.3.1+ to build the CFG of contracts. Particularly, the Ethersolve of CFG builder needs an environment of JDK 1.8+. Also, nowadays, the constraint-solving module is equipped with the Python versions of SMT libraries, including Z3 v4.12.1.0, Yices v2.6.4, and CVC4 v1.7. 

### Using Git

```bash
git clone https://github.com/Secbrain/VulHunter.git && cd VulHunter
```

Of course, VulHunter can also be deployed in a virtual environment. This project provides a virtual environment by default, under the venv folder. Execute the following command to start the virtual operating environment and use VulHunter directly. It is noted that this virtual environment works in Python 3.7, otherwise it needs to be recreated.

```bash
zip -F venv.zip --out venv_full.zip
unzip venv_full.zip
source venv/bin/activate
```

Note that the py-solc-x will use the solc bin files in the /home/user/.solcx (if user) or /root/.solcx (if root user), so we need to unzip the solc files to that path by executing the following command.

```bash
zip -F .solcx.zip --out solcx.zip
unzip -d /home/user solcx.zip
chmod -R 777 /home/user/.solcx/*
```

If you want to recreate the virtual environment, perform the following operations. Note that python3.7 needs to be replaced with the user's Python version.

```bash
virtualenv --python=/usr/bin/python3.7 venv
pip install torch==1.10.1
pip install tqdm
pip install scikit-learn==1.0.2
pip install pandas
pip install numpy==1.21.6
pip install pyevmasm==0.2.3
pip install py-solc-x==0.10.1
pip install evm_cfg_builder==0.3.1
pip install reportlab==3.6.12
pip install seaborn==0.12.2
pip install reportlab==3.6.12
pip install joblib==1.3.2 (for using traditional ML-based models)
```

Note that the optional verification module builds on the tool of [Manticore](https://github.com/trailofbits/manticore), thus users should install it by executing the following command before they invoke the verification function.

```bash
pip install -e ".[native]"
```

Also, the solvers such as Z3, CVC4, and Yices need to be installed based on the following operations, which are the same as the Manticore.

#### Installing Z3

Using pip to install the solver Z3.

```bash
pip install z3-solver==4.12.1.0
```

#### Installing CVC4

For more details go to https://cvc4.github.io/. Otherwise, just get the binary and use it.

```bash
sudo wget -O /usr/bin/cvc4 https://github.com/CVC4/CVC4/releases/download/1.7/cvc4-1.7-x86_64-linux-opt
sudo chmod +x /usr/bin/cvc4
```

#### Installing Yices

Yices is incredibly fast. More details here https://yices.csl.sri.com/

```bash
sudo add-apt-repository ppa:sri-csl/formal-methods
sudo apt-get update
sudo apt-get install yices2
```

## License

VulHunter is licensed and distributed under the AGPLv3 license.

## Publications

### References
- [ReJection: A AST-Based Reentrancy Vulnerability Detection Method](https://www.researchgate.net/publication/339354823_ReJection_A_AST-Based_Reentrancy_Vulnerability_Detection_Method), Rui Ma, Zefeng Jian, Guangyuan Chen, Ke Ma, Yujia Chen - CTCIS 19
- [DefectChecker: Automated Smart Contract Defect Detection by Analyzing EVM Bytecode](https://ieeexplore.ieee.org/document/9337195), Jiachi Chen, Xin Xia, David Lo, John Grundy, Xiapu Luo, Ting Chena, Yujia Chen - IEEE TSE
- [Smart Contract Vulnerability Detection using Graph Neural Network](https://www.ijcai.org/proceedings/2020/454), Yuan Zhuang, Zhenguang Liu, Peng Qian, Qi Liu, Xiang Wang, Qinming He - IJCAI 20
- [Slither: A Static Analysis Framework For Smart Contracts](https://arxiv.org/abs/1908.09878), Josselin Feist, Gustavo Grieco, Alex Groce - WETSEB '19
- [ETHPLOIT: From Fuzzing to Efficient Exploit Generation against Smart Contracts](https://wcventure.github.io/FuzzingPaper/Paper/SANER20_ETHPLOIT.pdf), Qingzhao Zhang, Yizhuo Wang, Juanru Li, Siqi Ma - SANER 20
- [SmartCheck: Static Analysis of Ethereum Smart Contracts](https://orbilu.uni.lu/bitstream/10993/35862/3/smartcheck-paper.pdf), Sergei Tikhomirov, Ekaterina Voskresenskaya, Ivan Ivanitskiy, Ramil Takhaviev, Evgeny Marchenko, Yaroslav Alexandrov - WETSEB '18

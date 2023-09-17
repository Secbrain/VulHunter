# SolversAdditional discussion

## The Limitaion of VulHunter

### Balance of model sequence space explosion and performance such as device computing power

There is an issue with the balance of analysis space and device restriction for the contract detection approaches, such as fuzzy testing, symbolic execution, and fuzzy testing. The VulHunter is no exception. Although it extracts some instances and then achieves better results, as mentioned in Section 4.7 of the paper, increasing the instance number can evaluate the contracts more comprehensively and further improve the detection performance. Therefore, in its practical usage, contract auditors can prefer the most appropriate settings to meet their analysis requirements by considering their equipment capacity and factors such as time overhead.

### The absence of contract datasets for new or uncommon vulnerabilities

The model cannot be trained when there are no datasets for the new or uncommon vulnerabilities. For these vulnerabilities, we can leverage the methods such as pattern matching and symbolic execution to define the logic of detection. Nonetheless, if the vulnerabilities are too complex to describe, the auditors can develop some similar and vulnerable contracts based on the limited vulnerability example to train the models. Also, due to the scalability of VulHunter, it may try to employ the traditional ML models and those based on Few-Shot or Zero-Shot learning, thus obtaining the models trained on the limited datasets. 

## The Improvement of VulHunter

### Vulnerability detection with multi-label

Vulnerability detection can be classified into binary categories (true or false), multi-category (true or one of vulunbilities), and multi-label (true or multiple vulunbilities) according to the number of vulnerability categories that can be detected at one time. Since a contract may incorporate multiple vulnerabilities, multi-category classification is unsuitable for our problem. Most SOTA methods (e.g., Oyente and Slither) are similar to VulHunter and belong to binary classification. That is, they all defined multiple detection patterns to detect varying vulnerabilities in contracts, and each pattern identifies the corresponding vulnerability. It is similar to how we invoke various classifiers. Also, they can all localize the vulnerabilities with specific lines in the contract source code. The difference is that our classifiers are trained on the dataset, and their patterns are pre-designed by experts. In addition, some methods leverage artificial intelligence (AI) to explore multi-label classification, such as ESCORT [CoRR_ESCORT]. They can consider multiple vulnerabilities simultaneously, which holds a superior detection efficiency than binary classification. Nonetheless, they mainly rely on assigning the threshold and are challenging to detect each vulnerability precisely. 
Fortunately, although VulHunter adopts a dichotomization design, the contract features are required to be extracted only once, which is similar to multi-label classification. Also, the inference of a single classifier is efficient (<10ms), and the time overhead of all classifiers can still be within an acceptable range. It is worth noting that classifiers can be analyzed in parallel, reducing the gap with multi-label classification. Nevertheless, the diversity of vulnerability categories makes VulHunter significant to consider multi-label detection in the real world. 

- [CoRR_ESCORT] Oliver Lutz, Huili Chen, Hossein Fereidooni, Christoph Sendner, Alexandra Dmitrienko, Ahmad Reza Sadeghi, Farinaz Koushanfar. [ESCORT: ethereum smart contracts vulnerability detection using deep neural network and transfer learning](https://arxiv.org/abs/2103.12607)[J]. arXiv preprint arXiv:2103.12607, 2021.

### Integration of useful expert knowledge

VulHunter is committed to the fully automatic identification of vulnerable sequences, that is, the whole process from training to detection without manual involvement. This can facilitate friendly use and avoid defective rules interfering with detection performance. Nevertheless, it cannot be ignored that precise and valuable expert knowledge can assist classifiers in identifying some vulnerabilities with prominent features (e.g., *reentrancy-eth*). This can guide the classifier in a beneficial direction based on blind learning. AME [IJCAI_Expert] and CGE [TKDE_GNN_Expert] were attempted in this regard. Although these arts can only work at the contract source code level, they motivate us to combine VulHunter with classical expert knowledge. 

- [IJCAI_Expert] Zhenguang Liu, Peng Qian, Xiang Wang, Lei Zhu, Qinming He, Shouling Ji. [Smart Contract Vulnerability Detection: From Pure Neural Network to Interpretable Graph Feature and Expert Pattern Fusion](https://www.ijcai.org/proceedings/2021/379)[C]//Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence. 2021: 2751--2759.
- [TKDE_GNN_Expert] Zhenguang Liu, Peng Qian, Xiaoyang Wang, Yuan Zhuang, Lin Qiu, Xun Wang. [Combining Graph Neural Networks With Expert Knowledge for Smart Contract Vulnerability Detection](https://ieeexplore.ieee.org/document/9477066)[J]. IEEE Transactions on Knowledge and Data Engineering, 2021.

## The Application Prospect of VulHunter

### Commercial/Product promotion

As security-as-a-service businesses are emerging, smart contract security and the contracting of these businesses is an interesting problem space. Similar to Slither, VulHunter can work with Remix to provide security audit services for the Ethereum contract developers. Also, it can serve as an extension plug-in (e.g., a component of the Blockchain Development Kit) for development tools such as VScode. In addition, the AI-based method can integrate the features of Webshell Backdoor to detect the security of website scripts (e.g., JSP and PHP). 
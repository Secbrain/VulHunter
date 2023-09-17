# Examples of security analysis reports

This folder includes three detailed contract analysis reports and three simplified contract analysis reports. From these two report types, we can know *the details of VulHunter's 31 security detectors*. Note that the difference between simplified and detailed is that the results of the detectors outputting secure are not displayed in the simplified report. That is, the simplified reports mainly focuses on briefing the contracts' problems, thereby reducing the time for auditors to read the reports. Nonetheless, vulnerabilities that do not exist in contracts illustrated in the detailed reports can alert owners to develop the contracts with specifications in the future. Besides, they are also classified into two types of detection reports with and without defective source code statements. The "map" version is the report with defective source code statements, otherwise, the report without defective source code statements.

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_opcodes_map.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_opcodes_map.pdf" is a detailed contract analysis report for contract *Revolution\_\_2* in Dataset_3. It includes five parts: summary information (e.g., contract name and audit time), contract audit results (e.g., vulnerability distribution and 31 of detectors' results), contract code (e.g., audit input and extracted instances), and contract audit details (i.e., the detailed results of 31 detectors, containing the vulnerable instances, defective source code statements, and the repair policy for contracts' vulnerabilities). 

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_opcodes_nomap.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_opcodes_nomap.pdf" is also a detailed contract analysis report for contract *Revolution\_\_2*. The difference between it and the above file is that there are no defective source code statements for vulnerable instances.

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_bytecodes_nomap.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_report_bytecodes_nomap.pdf" is also a detailed contract analysis report for contract *Revolution\_\_2*. The difference between it and the file "0xf3122a43ee86214e04b255ba78c980c43d0073e2\_Revolution\_\_2\_report\_opcodes\_nomap.pdf" is that the vulnerable instances in this file are bytecode sequences.

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_opcodes_map.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_opcodes_map.pdf" is a simplified contract analysis report for contract *Revolution\_\_2*. Its contents are similar to the file "0xf3122a43ee86214e04b255ba78c980c43d0073e2\_Revolution\_\_2\_report\_opcodes\_map.pdf", except that it does not describe the recommendations of the detectors that output secure.

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_opcodes_nomap.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_opcodes_nomap.pdf" is a simplified contract analysis report for contract *Revolution\_\_2*. Its contents are similar to the file "0xf3122a43ee86214e04b255ba78c980c43d0073e2\_Revolution\_\_2\_report\_bytecodes\_nomap.pdf", except that it does not describe the recommendations of the detectors that output secure.

## 0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_bytecodes_nomap.pdf

The file "0xf3122a43ee86214e04b255ba78c980c43d0073e2_Revolution__2_main_report_bytecodes_nomap.pdf" is also a simplified contract analysis report for contract *Revolution\_\_2*. The difference between it and the file "0xf3122a43ee86214e04b255ba78c980c43d0073e2\_Revolution\_\_2\_main\_report\_opcodes\_nomap" is that the vulnerable instances in this file are bytecode sequences.

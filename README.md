# BioDefect: The First Dataset for Defect Detection in Bioinformatics Software

## 🚀 Overview
**BioDefect** is the first dataset specifically designed for defect detection in bioinformatics software. Existing defect detection datasets, such as **Devign** and **REVEAL**, fail to generalize to bioinformatics due to the unique characteristics of bioinformatics code. 

---

## 📂 Repository Structure
```bash
BioDefect/
│── 📁 Dataset/                     # Contains datasets used in the study, including BioDefect
│   ├── 📂 BioDefect/               # The BioDefect dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 Scanpy_test.jsonl    # Testing dataset
│   │   ├── 📜 Bowtie2_test.jsonl   # Testing dataset
│   │   ├── 📜 BWA_test.jsonl       # Testing dataset
│   │   ├── 📜 Details.xlsx         # Detailed information about defect functions
│   │   └── ...
│   ├── 📂 Devign/                  # Existing dataset used for comparison
│   └── 📂 REVEAL/                  # Existing dataset used for comparison
│
│── 📁 Defect_detection/            # Implementations of defect detection models
│   ├── 🤖 Test1_bert/              # BERT model implementation
│   │   ├── 📜 clss_indices.json    # Label mapping file
│   │   ├── 📜 model.py             # Model definition
│   │   ├── 📜 run.py               # Script for fine-tuning the model
│   │   ├── 📜 test.py              # Script for model evaluation
│   │   └── ...
│   ├── 🤖 Test2_codebert/          # CodeBERT model implementation
│   ├── 🤖 Test3_t5/                # T5 model implementation
│   ├── 🤖 Test4_codet5/            # CodeT5 model implementation
│   ├── 🤖 Test5_codet5+/           # CodeT5+ model implementation
│   ├── 🤖 Test6_opt/               # OPT model implementation
│   ├── 🤖 Test7_codegen/           # CodeGen model implementation
│   ├── 🤖 Test8_deepseek/          # DeepSeek-R1 model implementation
│   └── 🤖 Test9_starcoder2/        # StarCoder2 model implementation
│   
│── 📜 environment.yaml             # Environment configuration file
│── 📜 results.xlsx                 # Detailed results from the study
│── 📜 README.md                    
└── ...
```

---

## 📊 Dataset
Existing defect detection datasets are mainly constructed from large-scale, mature software engineering projects, which differ significantly from bioinformatics software in terms of **programming languages, coding standards, and defect patterns**. BioDefect, built from real bioinformatics software, integrates these characteristics to enhance the precision of defect detection in this domain.
- ✅ **Programming Languages:** Due to its specialized application scenarios, bioinformatics software is predominantly developed in Python (for data processing and analysis) and R (for statistical analysis and visualization). However, most widely used defect detection datasets focus on C/C++ and Java. This discrepancy in programming languages suggests that existing datasets may not be well-suited for detecting defects in bioinformatics software. Additionally, some bioinformatics software projects employ multi-language development, further complicating defect detection when using existing datasets.
- ✅ **Coding Conventions:** Bioinformatics software is often developed by academic researchers, many of whom lack formal software engineering backgrounds. As a result, bioinformatics software frequently exhibits diverse coding styles, non-standard practices, and a lack of clear module separation. These irregularities make it difficult for existing datasets to effectively address the defects present in such unstructured code.
- ✅ **Defect Patterns:** Defects in bioinformatics software go beyond common code errors and extend to algorithmic errors that can lead to incorrect biological inferences or inaccurate computational results. For example, an error in genomic sequence alignment might not cause a runtime failure or crash, yet it could yield misleading biological conclusions. Existing datasets primarily focus on conventional software defects (e.g., security vulnerabilities, memory issues), which may be insufficient for detecting the unique types of defects encountered in bioinformatics software.

BioDefect consists of a primary training set and three independent test sets, including the entire source code repository containing defective code, making it more representative of real-world bioinformatics software defects. This ensures comprehensive and precise model evaluation across various scenarios. Additionally, BioDefect addresses label inconsistency through manual verification and improved data collection strategies while mitigating data leakage using a software time-series approach.

Each sample in BioDefect includes the following information:
- ✅**Project**: The name of the project the sample belongs to.
- ✅**Commit ID**: The commit version from which the function was extracted.
- ✅**Target**: Label information, where `0` represents non-defective code and `1` represents defective code.
- ✅**Func**: The source code of the function.
- ✅**Idx**: The sample index number.

To facilitate future research, we provide detailed defect function information in `Dataset/BioDefect/details.xlsx`.

---

## 🧠 Defect Detection Models
This repository also provides implementations of various language models evaluated in our study, including:
- 🤖 **BERT**
- 🤖 **CodeBERT**
- 🤖 **T5**
- 🤖 **CodeT5**
- 🤖 **CodeT5+**
- 🤖 **OPT**
- 🤖 **CodeGen**
- 🤖 **DeepSeek-R1**
- 🤖 **StarCoder2**

Each model's directory contains:
- 📜 `model.py` - Model definition
- 📜 `run.py` - Fine-tuning script
- 📜 `test.py` - Evaluation script
- 📜 `class_indices.json` - Label mapping file

---

## 💻 Experiments
### 📥 Install
```sh
conda env create -f environment.yml
```

### 📂 Using the Dataset
You can access the dataset under the `Dataset/` directory. For more details, please refer to our paper.

### 🚀 Running Defect Detection Models
Each model has its own directory under `Defect_detection/`. To run a specific model, navigate to its directory and follow the provided instructions.

#### Example: Fine-tuning DeepSeek-R1
```sh
cd Defect_detection/Test8_deepseek
python run.py \
    --train_data_file=../../dataset/BioDefect/train.jsonl \
    --eval_data_file=../../dataset/BioDefect/valid.jsonl \
    --output_dir=./saved_models \
    --runs_path=./runs \
    --model_type=deepseek \
    --tokenizer_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --do_train \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --learning_rate 1e-6 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1
    --adam_epsilon 1e-8
    --evaluate_during_training \
    --seed 123456
```

#### Example: Evaluation
```sh
cd Defect_detection/Test8_deepseek
python test.py \
    --test_data_file=../../dataset/BioDefect/Scanpy_test.jsonl \
    --output_dir=./saved_models \
    --results_path=./results \
    --model_type=deepseek \
    --tokenizer_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --do_test \
    --block_size 400 \
    --eval_batch_size 8 \
    --seed 123456
```

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

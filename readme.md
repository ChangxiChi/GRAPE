## GRAPE: Heterogeneous Graph Representation Learning for Genetic Perturbation with Coding and Non-Coding Biotype (IJCAI2025)
## 🏁 Getting Started

### Preparation
The dataset can be downloaded from the following URL:
```bash
Adamson: https://dataverse.harvard.edu/api/access/datafile/6154417
Norman: https://dataverse.harvard.edu/api/access/datafile/6154020
```
The pre-trained LLMs can be downloaded from the reference mentioned in the main text.

## 📁 Repository Structure  
```plaintext
GRAPE/
├── data/                       # Dictionary, create it and download dataset to here
├── from_pertrain_bert/         # Pre-trained LLM for gene descriptions
├── from_pertrain_DNA_hyena/    # Pre-trained LLM for DNA sequences
├── results/                    # Output 
│   ├── DatasetName_HvgNum/     # Results
├── test.py
├── model.py
└── ...          
```

### Step 1: Run Preprocess.py - Obtain Gene Information from Ensembl Database  
In the first step, run this file to fetch gene-related information from the Ensembl database using its API. This will retrieve gene descriptions and biotype data necessary for downstream tasks.  
```bash
python Preprocess.py --output result/DatasetName_HvgNum/
```

### Step 2: Run extract_embed.py - Extract Gene Representations Using Pre-trained LLM  
Next, run this file to extract feature representations from the gene descriptions and DNA sequences using pre-trained LLMs. These representations will serve as the initialization for gene representations in the model.
```bash
python extract_embed.py --output result/DatasetName_HvgNum/
```

### Step 3: Run test.py - Train GRAPE 
Finally, run this file to train our model.
```bash
python test.py --output result/DatasetName_HvgNum/
```

### Step 4: Evaluation Results
See 📁 [`predict.py`](./predict.py) for details.
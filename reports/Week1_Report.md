# Week 1 Progress Report

# Project Title

**Offline Log Analysis using Small Language Models (SLM)**

---

# Week

**Week 1 – Environment Setup and Offline Preparation**

---

# Objective

The objective of Week 1 was to establish a fully functional offline development environment for the Log Analysis using Small Language Models (SLM) project.

Since the target deployment environment does not allow internet connectivity, every dependency, machine learning model, tokenizer, dataset, and supporting library needed to be downloaded, verified, and organized during this phase. The environment was then validated in a clean virtual environment to ensure complete reproducibility before proceeding to Week 2.

---

# Environment Details

| Component          | Details                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| Operating System   | Windows                                                                 |
| Python Environment | Virtual Environment (venv)                                              |
| Package Manager    | pip                                                                     |
| Version Control    | Git                                                                     |
| Model              | Qwen2.5-3B-Instruct                                                     |
| Embedding Model    | all-MiniLM-L6-v2                                                        |
| Dataset            | LogHub                                                                  |
| Frameworks         | PyTorch, Transformers, Sentence Transformers, FAISS, FastAPI, Streamlit |

---

# Task 1 – Create Python Virtual Environment

A dedicated virtual environment was created to isolate project dependencies from the system Python installation.

## Command Executed

```powershell
python -m venv venv
```

## Activate Environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

## Verification

The terminal prompt changed to:

```text
(venv)
```

indicating successful activation.

---

# Task 2 – Download All Offline Python Packages

Since package installation will not be possible in the offline deployment environment, all required Python packages were downloaded locally.

## Commands Executed

```powershell
pip download torch torchvision torchaudio --dest ./offline_packages

pip download transformers datasets peft accelerate --dest ./offline_packages

pip download sentence-transformers faiss-cpu --dest ./offline_packages

pip download fastapi uvicorn streamlit --dest ./offline_packages
```

During verification it was observed that the tokenizer dependencies required by the Qwen model were missing.

Additional packages were therefore downloaded.

```powershell
pip install sentencepiece tiktoken

pip download sentencepiece tiktoken --dest ./offline_packages
```

## Packages Collected

* torch
* torchvision
* torchaudio
* transformers
* datasets
* peft
* accelerate
* sentence-transformers
* faiss-cpu
* fastapi
* uvicorn
* streamlit
* sentencepiece
* tiktoken

along with all their required dependencies.

---

# Task 3 – Prepare Requirements File

A project requirements file was prepared for future offline installation.

## requirements.txt

```text
torch
torchvision
torchaudio
transformers
datasets
peft
accelerate
sentence-transformers
faiss-cpu
fastapi
uvicorn
streamlit
```

A second file named **requirements_verified.txt** was created to store the exact installed package versions after successful verification.

---

# Task 4 – Download Qwen2.5-3B-Instruct Model

The Qwen2.5-3B-Instruct language model was downloaded while internet access was available.

## Command Executed

```powershell
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', cache_dir='./models/qwen25-3b')"
```

The model was stored locally under

```text
models/qwen25-3b/
```

---

# Task 5 – Download Sentence Transformer Model

The embedding model required for semantic search was downloaded.

## Command Executed

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/sentence-transformers')"
```

The model was successfully cached inside

```text
models/sentence-transformers/
```

---

# Task 6 – Download Dataset

The LogHub dataset repository was cloned into the project.

Repository location:

```text
data/loghub/
```

The dataset contains benchmark logs from:

* Android
* Apache
* BGL
* HDFS
* Hadoop
* HealthApp
* Linux
* Mac
* OpenSSH
* OpenStack
* Proxifier
* Spark
* Thunderbird
* Windows
* Zookeeper

These datasets will be used during parser development and model evaluation.

---

# Task 7 – Verify Model Downloads

Initially, loading the tokenizer using the cache root resulted in an error because Hugging Face stores downloaded models inside snapshot directories.

The tokenizer was successfully verified using the snapshot path.

## Command Executed

```powershell
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('./models/qwen25-3b/models--Qwen--Qwen2.5-3B-Instruct/snapshots/<snapshot-id>')"
```

## Verification Result

Tokenizer loaded successfully without errors.

---

# Task 8 – Create Clean Offline Test Environment

To simulate the final offline deployment environment, a fresh virtual environment was created.

## Commands Executed

```powershell
python -m venv testenv

.\testenv\Scripts\Activate.ps1
```

---

# Task 9 – Verify Offline Installation

The complete project environment was installed exclusively from the local offline package repository.

## Command Executed

```powershell
pip install --no-index --find-links ./offline_packages -r requirements.txt
```

Installation completed successfully without accessing the internet.

---

# Task 10 – Verify Python Packages

The major packages were imported successfully.

## Command Executed

```powershell
python -c "import torch, transformers, datasets, sentence_transformers, fastapi, streamlit; print('Offline packages verified.')"
```

Output

```text
Offline packages verified.
```

---

# Task 11 – Verify Qwen Tokenizer

## Command Executed

```powershell
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('./models/qwen25-3b/models--Qwen--Qwen2.5-3B-Instruct/snapshots/<snapshot-id>'); print('Qwen verified.')"
```

Output

```text
Qwen verified.
```

---

# Task 12 – Verify Sentence Transformer

## Command Executed

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/sentence-transformers'); print('Sentence Transformer verified.')"
```

Output

```text
Sentence Transformer verified.
```

---

# Task 13 – Verify Additional Tokenizer Libraries

## Command Executed

```powershell
pip install --no-index --find-links ./offline_packages sentencepiece tiktoken

python -c "import sentencepiece, tiktoken; print('Extra packages verified.')"
```

Output

```text
Extra packages verified.
```

---

# Task 14 – Git Repository Initialization

The project repository was initialized using Git.

## Commands Executed

```powershell
git init

git status
```

A `.gitignore` file was created to exclude:

* virtual environments
* cached files
* model weights
* offline package bundles
* environment variables
* IDE configuration

Contents:

```text
venv/
testenv/
__pycache__/
*.pyc

models/
offline_packages/

.env

.vscode/
```

---

# Task 15 – Resolve Embedded Git Repository

The downloaded LogHub dataset contained its own Git repository.

To ensure that all dataset files are tracked by the main project repository, the embedded Git metadata was removed.

## Commands Executed

```powershell
Remove-Item -Recurse -Force .\data\loghub\.git

Test-Path .\data\loghub\.git
```

Output

```text
False
```

The dataset was then staged successfully.

```powershell
git add data/loghub
```

---

# Challenges Encountered

## Missing Tokenizer Dependencies

While verifying the Qwen tokenizer, an error indicated that the required tokenizer backends were unavailable.

### Resolution

Installed and downloaded:

* sentencepiece
* tiktoken

for future offline use.

---

## Incorrect Tokenizer Path

The tokenizer initially failed to load because the Hugging Face cache root was used instead of the snapshot directory.

### Resolution

Loaded the tokenizer using the snapshot path generated by Hugging Face.

---

## Embedded Git Repository

The LogHub dataset contained an internal `.git` folder.

### Resolution

The nested Git metadata was removed before staging the dataset.

---

# Deliverables Completed

* Python virtual environment
* Offline dependency bundle
* requirements.txt
* requirements_verified.txt
* Qwen2.5-3B-Instruct model
* Sentence Transformer model
* LogHub dataset
* Offline installation verification
* Git repository initialization
* Project documentation

---

# Outcome

Week 1 was successfully completed.

A fully reproducible and offline-ready development environment has been established. All required dependencies, language models, embedding models, datasets, and supporting libraries have been downloaded, verified, and organized. The setup was validated using a separate clean virtual environment to ensure that the project can be installed and executed entirely offline.

The project is now ready for **Week 2**, where implementation of the log parsing, embedding generation, retrieval pipeline, and Small Language Model integration will begin.



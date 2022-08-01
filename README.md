# CrystalBLEU

## Install
Install the requirements:
```bash
pip install -r requirements.txt
```
Place `bleu_ignoring.py` accessible to your python script.

## Usage
```python
from collections import Counter
# Import CrystalBLEU
from bleu_ignoring import corpus_bleu

# Extract trivially shared n-grams
k = 500
frequencies = Counter(tokenized_corpus)
trivially_shared_ngrams = dict(frequencies.most_common(k))

# Calculate CrystalBLEU
crystalBLEU_score = corpus_bleu(
    references, candidates, ignoring=trivially_shared_ngrams)
```

---------------------------

## Reproducing Paper Results
The `scripts` directory contains scripts that generate results shown in the paper, based on the figure or table number, or the dataset used.  
To reproduce the results from the paper run the following scripts in order:
```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare the datasets
bash ./scripts/prepare_data.sh english french java python c

# Print the data in Table 1 and plot Figure 3
bash ./scripts/table1-figure3.sh

# Plot Figure 4
bash ./scripts/figure4.sh

# Print the first two rows of Table 2
bash ./scripts/table2.sh

# Print results from BigCloneBench:
# - the third row of Table 2, 
# - the right columns of Table 3,
# - rows 3 & 4 of Table 4,
# - the third row of Table 5, and
# - Figure 5
bash ./scripts/big_clone_bench.sh

# Print results from ShareCode:
# - the left columns of Table 3,
# - rows 1 & 2 of Table 4, and
# - rows 1 & 2 of Table 5
bash ./scripts/sharecode.sh

# Results from Concode:
# - the last row of Table 4,
# - the last row of Table 5, and
# - plot Figure 6
bash ./scripts/figure6.sh

# Plot Figure 7
bash ./scripts/figure7.sh

# Plot Figure 8
bash./scripts.figure8.sh
```


When running `scripts/prepare_data.sh`, you can select any subset of `english`, `french`, `java`, `python`, and `c`. This allows downloading and preparing only the selected datasets. Note: this might cause some scripts to crash if they cannot find the required data files.  
Scripts are mostly independent of eachother, except for the following:  
- To run `figure8.sh`, you need to first run `big_clone_bench.sh` and `sharecode.sh`.

---------------------------

## Contents
- `CodeBLEU/` contains the implementation of [CodeBLEU](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU), obtained from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).
- `clone_detection/` contains the BigCloneBench dataset, obtained from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).
- `concode/` contains the Concode dataset, obtained from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).
- `sc_clone/` is derived from the [ShareCod](https://sharecode.io/) dataset in the style of BigCloneBench.
- `scripts/` contains the scripts needed to reproduce the results from our paper.
- `extra/` contains the scripts that were not used in the paper.
- `ase2022-paper108.pdf` is a copy of our ASE'22 accepted submission.
- `lang{0, 1, 2}.json` are data files from ShareCode for each of `C`, `C++`, and `Java` languages.

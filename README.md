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
The `scripts` directory contains scripts that generates results shown in the paper, based on the figure or table number, or the dataset used.  
To run the experiments, first download and prepare the datasets by running:
```bash
bash ./scripts/prepare_data.sh english french java python c
```
You can select any subset of `english`, `french`, `java`, `python`, and `c`.  
Scripts are mostly independent of eachother, except for the following:  
- To run `figure8.sh`, you need to first run `big_clone_bench.sh` and `sharecode.sh`.

---------------------------

## Data
- `scores_*.json` contain the human study score results
- `data*.json` contain the code pairs of the human study (the ids match with scores)

## Scripts
- The implementation of CrystalBLEU is available in `bleu_ignoring.py`
- `prepare_human_study.py` generates the pairs of programs for the human study
- `analyze_results.py` outputs the correlations regarding the human study
- `small_v1.py` calculates the distinguishability for ShareCode dataset
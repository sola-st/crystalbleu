# CrystalBLEU

## Data
- `scores_*.json` contain the human study score results
- `data*.json` contain the code pairs of the human study (the ids match with scores)

## Scripts
- The implementation of CrystalBLEU is available in `bleu_ignoring.py`
- `prepare_human_study.py` generates the pairs of programs for the human study
- `analyze_results.py` outputs the correlations regarding the human study
- `small_v1.py` calculates the distinguishability for ShareCode dataset

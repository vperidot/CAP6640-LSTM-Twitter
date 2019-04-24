# Generating Tweets with Long Short-Term Memory Networks

files:
* `nlp_preprocessing.py` : contains data and preprocessing functions
* `model.py` : contains LSTM model creation functions
* `train.py` : contains training and evaluation functions
* `experiments_*.py` : contains code to run experiments

## Experiment Results

### Network Depth
**Word Model**

| Depth | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | LOSS  | ACC    |
|-------|--------|--------|--------|--------|-------|--------|
| 1     | 0.3441 | 0.2419 | 0.2120 | 0.1897 | 5.801 | 0.1425 |
| 2     | 0.3218 | 0.2258 | 0.1940 | 0.1725 | 6.081 | 0.1304 |
| 3     | 0.2768 | 0.2169 | 0.1952 | 0.1788 | 6.066 | 0.1045 |

**Character Model**

| Depth | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | LOSS  | ACC    |
|-------|--------|--------|--------|--------|-------|--------|
| 1     | 0.1967 | 0.1208 | 0.0891 | 0.0766 | 1.478 | 0.5612 |
| 2     | 0.1963 | 0.1256 | 0.0916 | 0.0782 | 1.405 | 0.5831 |
| 3     | 0.1944 | 0.1255 | 0.0938 | 0.0821 | 1.385 | 0.5939 |

### Dropout (regularization)
**Word Model**
Note: the first row (dropout=0) corresponds to the Depth=1 experiment above.

| Dropout | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | LOSS  | ACC    |
|---------|--------|--------|--------|--------|-------|--------|
| 0       | 0.3441 | 0.2419 | 0.2120 | 0.1897 | 5.801 | 0.1425 |
| .25     | 0.2912 | 0.2068 | 0.1808 | 0.1607 | 5.475 | 0.1560 |
| .50     | 0.2962 | 0.2244 | 0.1986 | 0.1786 | 5.269 | 0.1693 |
| .65     | 0.2744 | 0.2158 | 0.1921 | 0.1727 | 5.273 | 0.1627 |

**Character Model**

### Selected Model

**Donald Trump**
* Metrics
* Sample Outputs

**Michelle Obama**
* Metrics 
* Sample Outputs

**Ellen**
* Metrics 
* Sample Outputs

# AutoDIP

Preferential HPO for [chtorch](https://github.com/dhis2-chap/chtorch/tree/master)

## Getting Started

### Installing AutoDIP/chtorch

- clone this repo
- create venv (ideally using Python 3.11.3)
- `pip install -e .`

### Adding pbmohpo

- clone [pbmohpo](https://github.com/ja-thomas/pbmohpo/tree/main) inside the AutoDIP folder
- cd into the pbmohpo folder
- `pip install -e ".[experiments]"`

### Solving dependency issues

Without these steps, there is a mismatch between python version, numpy version and ConfigSpace version.

- `pip install ConfigSpace==0.6.1`

This leads to the error 
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
pbmohpo 0.1 requires ConfigSpace==0.6.0, but you have configspace 0.6.1 which is incompatible. 
```

which can be ignored.

- `pip install numpy~=1.26.4`

### Running code

HPO using optuna:

`python scripts/auto_hpo.py data/vietnam_monthly.csv `

Preferential HPO using pbmohpo:

`python scripts/preference_based_auto_hpo.py data/vietnam_monthly.csv`
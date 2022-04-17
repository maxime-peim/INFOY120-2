# INFOY-120 Project 2

## Install manipulation module

```bash
$ python3.9 -m venv env
$ source env/bin/activate
$ python3 -m pip install -r requirements.txt
```

## Extract and preprocess EML files

```bash
$ python3 main.py extract preprocess
# see the result in data/TR and data/TT
```

## Evaluate classifers and parameters from main.py and predict

```bash
$ python3 main.py classify
# see the result in data/TT/labels
```
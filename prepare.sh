#!/bin/bash

sudo apt install python3.9 python3.9-venv
python3.9 -m venv env
source env/bin/activate

pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
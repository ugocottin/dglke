#!/bin/bash
python3 -m pip install --user virtualenv
python3 -m venv env
source .venv/bin/activate
pip install pysmiles
#!/bin/sh

git clone https://github.com/pdollar/coco

# API install
cd coco/PythonAPI
# If you use the python 3, you have to use bellow command
2to3 -w {your python 3 library}/cocotools/*.py

python setup.py install
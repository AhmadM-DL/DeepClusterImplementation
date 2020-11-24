#!/bin/bash

LRs=seq 0.1 2 10
WD=0.00001
K=10
PYTHON="C:\ProgramData\Anaconda3\python.exe"

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
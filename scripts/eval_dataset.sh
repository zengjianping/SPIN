#!/bin/bash

DATASET=h36m-p2

python3 eval.py \
    --checkpoint=data/model_checkpoint.pt \
    --dataset=$DATASET --log_freq=20



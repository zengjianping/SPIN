#!/bin/bash

DATASET=3dpw

python3 eval.py \
    --checkpoint=data/model_checkpoint.pt \
    --dataset=$DATASET --log_freq=20



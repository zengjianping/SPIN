#!/bin/bash

python3 train.py \
    --name train_example \
    --pretrained_checkpoint=data/model_checkpoint.pt \
    --run_smplify


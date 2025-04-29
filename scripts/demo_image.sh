#!/bin/bash

python demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json
python demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --bbox=examples/im1010_bbox.json
python demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg


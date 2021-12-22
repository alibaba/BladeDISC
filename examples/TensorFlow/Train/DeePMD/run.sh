#!/bin/bash

export DP_INTERFACE_PREC=low
export CUDA_VISIBLE_DEVICES=0

python main.py train input.json

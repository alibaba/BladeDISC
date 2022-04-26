#!/bin/bash

# This script copies headers and libraries of cuDNN from /usr to /usr/local/cuda in
# nvidia:cuda-10.2-xxx images. This is needed because tensorflow assumes that.

[[ $1 != "cu102" ]] && { echo "Skip pathcing for non-cu102 images."; exit 0; }

headers=$(ls /usr/include/*.h | grep -E 'cublas|cudnn|nvblas')
for hdr in ${headers[@]}; do
    echo "Copy ${hdr} to cuda home."
	cp ${hdr} /usr/local/cuda/include/
done

libs=$(ls /usr/lib/x86_64-linux-gnu/ | grep -E 'cublas|cudnn|nvblas' | grep -E '\.so|\.a')
for lib in ${libs[@]}; do
    echo "Copy /usr/lib/x86_64-linux-gnu/${lib} to cuda home."
	cp /usr/lib/x86_64-linux-gnu/${lib} /usr/local/cuda/lib64/
done

# Introduction

This directory contains some advanced quantization algorithms. These quantization algorithms are 
difficult to implement under the original torch-quant framework (or not necessary, for example, 
if you do weight-only quantization, fx graph may be not necessary).


# Supported algorithms

### GPTQ
The official [GPTQ codes](https://github.com/IST-DASLab/gptq) are referenced.
```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```

# Introduction

This directory contains some advanced quantization algorithms. These quantization algorithms are 
difficult to implement under the original torch-quant framework (or not necessary, for example, 
if you do weight-only quantization, fx graph may be not necessary). So under the premise of ensuring
that the interface is consistent with that in torch-quant, we implement the corresponding quantizer
for each advanced quantization algorithm alone.


# Supported algorithms

### GPTQ
The official [GPTQ codes](https://github.com/IST-DASLab/gptq) are referenced.

NOTE: There is one small difference between the official GPTQ implementation and the one here.
In the official implementation, the inputs used to calculate the H matrix of a specific layer,
are obtained after the weight of the previous layers are quantized, which is not easy to be achieved
(The model should be calibrated many times). We relex this condition, that is, the inputs used to
calculate the H matrix are obtained when the weight of the previous layers are NOT quantized. So
the calibration data only needs to be executed once on the model, and all H matrices can be calculated.
This relaxation only results in a slight loss of performance for 4 bit quantization. Of course, we are
also figuring out how to enable this in a user-friendly way.

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```

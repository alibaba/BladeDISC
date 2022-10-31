# BladeDISC Quickstart

BladeDISC developed TensorFlow and PyTorch wrapper to make users easier
to improve machine learning performance on native TensorFlow and PyTorch
program.

This document introduced a quick and simple demo of BladeDISC.  Please
make sure you have read [Install BladeDISC with Docker](./install_with_docker.md).

## Quickstart for TensorFlow Users

TensorFlow Blade provides a simple Python API with just **TWO LINES** of codes
on native TensorFlow program as the following:

``` python
import blade_disc_tf as disc
disc.enable()
```

It is recommended to fetch the latest BladeDISC runtime Docker image
with TensorFlow for a smooth setup:

``` bash
docker pull bladedisc/bladedisc:latest-runtime-tensorflow1.15
```

A simple demo is as the following:

``` python
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# enable BladeDISC with the following two lines!
import blade_disc_tf as disc
disc.enable()


g = tf.Graph()
with g.as_default():
    # reduce_sum((x + y) * c) ^ 2
    x = tf.placeholder(shape=[None, None], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None, None], dtype=tf.float32, name="y")
    c = tf.constant([[1], [2]], dtype=tf.float32, shape=(2, 1), name="c")
    t1 = x + y
    t2 = tf.matmul(t1, c)
    t3 = tf.reduce_sum(t2)
    ret = t3 * t3

    with tf.Session() as s:
        np_x = np.ones([1, 2]).astype(np.float32)
        np_y = np.ones([1, 2]).astype(np.float32)
        r = s.run(ret, {x: np_x, y: np_y})
        print("x.shape={}, y.shape={}, ret={}".format(np_x.shape, np_y.shape, r))
```

Read more [TensorFlow tutorial](./tutorials/tensorflow_inference_and_training.md).

## Quickstart for PyTorch Users

To make PyTorch users easier to use, BladeDISC provides simple
Python API is as follows:

``` python
import torch_blade

with torch.no_grad():
    # blade_model is the optimized module by BladeDISC
    blade_model = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(inputs))
```

It is recommended to fetch the latest runtime Docker image with PyTorch
for a smooth setup:

``` bash
docker pull bladedisc/bladedisc:latest-runtime-torch1.7.1
```

`torch_blade` accepts an `nn.Module` object and outputs the optimized module,
a quick demo is as follows:

``` python
import torch
import torch_blade
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.c = torch.randn(10, 3)

    def forward(self, x, y):
        t1 = x + y
        t2 = torch.matmul(t1, self.c)
        t3 = torch.sum(t2)
        return t3 * t3

my_cell = MyCell()
x = torch.rand(10, 10)
h = torch.rand(10, 10)

with torch.no_grad():
    blade_cell = torch_blade.optimize(my_cell, allow_tracing=True, model_inputs=(x, y))

print(blade_cell(x, h))
```

Read more [PyTorch tutorial](./tutorials/torch_bert_inference.md).

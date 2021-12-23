# Quick Start for TensorFlow Wrapper

This tutorial is for a quick try with BladeDISC, please make sure you have
[installed Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on your host.

To enable BladeDISC, only **TWO LINES** of codes are needed on native Tensorflow
program as the following:

``` python
import blade_disc_tf as disc
disc.enable()
```

It is recommended to fetch the latest BladeDISC runtime Docker image:
`yancey1989/bladedisc:latest-runtime-tf115` for a smooth setup.

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
        for i in range(10, 20, 5):
            np_x = np.ones([i, i]).astype(np.float32)
            np_y = np.ones([i, i]).astype(np.float32)
            r = s.run(ret, {x: np_x, y: np_y})
            print("x.shape={}, y.shape={}, ret={}".format(np_x.shape, np_y.shape, r))
```
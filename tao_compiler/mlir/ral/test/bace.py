#!/usr/bin/env python
# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import print_function

import os, sys
import numpy as np

import tensorflow.compat.v1 as tf
#import tensorflow as tf
tf.disable_v2_behavior()
from tensorflow.core.framework import attr_value_pb2

def main():
  print('Hello')

  to_np_dtype = {tf.float32: np.float32, tf.float16: np.float16}
  np.random.seed(1)

  #m, n, k = 258, 511, 1023
  m, n, k = 2, 5, 3
  shape1 = (m, k)
  shape2 = (n, k)
  dtype = tf.float32
  tp_a = False
  tp_b = False

  a = tf.placeholder(name='a', shape=[None, None], dtype=dtype)
  b = tf.placeholder(name='a', shape=[None, None], dtype=dtype)
  if (tp_b):
    kernel = np.random.random_sample([n, k]).astype(to_np_dtype[dtype])
  else:
    kernel = np.random.random_sample([k, n]).astype(to_np_dtype[dtype])


  scope = "jit_scope_tao_" + str(0)
  attrs = {
    "_XlaCompile": attr_value_pb2.AttrValue(b=True),
    "_XlaSeparateCompiledGradients": attr_value_pb2.AttrValue(b=False),
    "_XlaScope": attr_value_pb2.AttrValue(s=scope.encode())
  }
 
  with tf.get_default_graph()._attr_scope(attrs):
    #b = tf.constant(kernel)
    c = tf.matmul(a, b, transpose_a=tp_a, transpose_b=tp_b)

  if (tp_a):
    pa = np.random.random_sample([k, n]).astype(to_np_dtype[dtype])
  else:
    pa = np.random.random_sample([m, k]).astype(to_np_dtype[dtype])


  sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, log_device_placement=True))

  for i in range(10):
    r = sess.run([c], {a : pa, b : kernel})
    print('iter #{}'.format(i), r)

  sess.close()

  pass


if __name__ == '__main__':
  main()

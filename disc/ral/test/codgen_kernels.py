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

  shape = (2560,3780)
  shape = (2,3)
  dtype = tf.float32

  px = tf.placeholder(name='x', shape=shape, dtype=dtype)
  py = tf.placeholder(name='y', shape=shape, dtype=dtype)
  pz = tf.placeholder(name='z', shape=shape, dtype=dtype)

  scope = "jit_scope_tao_" + str(0)
  attrs = {
    "_XlaCompile": attr_value_pb2.AttrValue(b=True),
    "_XlaSeparateCompiledGradients": attr_value_pb2.AttrValue(b=False),
    "_XlaScope": attr_value_pb2.AttrValue(s=scope.encode())
  }
 
  with tf.get_default_graph()._attr_scope(attrs):
    t1 = px + py
    t2 = py + pz
    t3 = t1 + t2
    #t4 = tf.reduce_sum(t3, axis=1, keepdims=True, name='reduce')
    t4 = px + pz
    t4 = t4 + t4

  sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, log_device_placement=True))

  np.random.seed(1)
  np_x = np.random.rand(*shape)
  np_y = np.random.rand(*shape)
  np_z = np.random.rand(*shape)

  for i in range(10):
    r = sess.run([t3, t4], {px : np_x , py : np_y, pz : np_z})
    #r = sess.run([t3], {px : np_x , py : np_y, pz : np_z})
    print('iter #{}'.format(i), r)

  sess.close()

  pass


if __name__ == '__main__':
  main()

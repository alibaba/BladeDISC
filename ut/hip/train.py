#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Alibaba Inc.
# File              : train.py
# Author            : Yue Wu <matthew.wy@alibaba-inc.com>
# Date              : 2022-03-17
# Last Modified Date: 2022-03-17
# Last Modified By  : Yue Wu <matthew.wy@alibaba-inc.com>
# import tensorflow. as tf
from numpy.core.fromnumeric import transpose
#import disc_dcu
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import click
import os, time, shutil
import argparse 


class NetworkMem():
    def __init__(self, x0, x1, dtype=tf.float32, use_const=False):
        if use_const:
            x_data = np.linspace(-1,1, x0* x1).reshape(x0,x1)
            self.x = tf.constant(x_data)
        else:
            self.x = tf.placeholder(shape=[x0, x1],dtype=dtype)
        self.x0 = tf.add(self.x, self.x)
        self.x1 = tf.multiply(self.x0, self.x)
        self.x2 = tf.multiply(self.x1, 0.5)
        self.output = tf.add(self.x2, 0.5)

class NetworkSoft():
    def __init__(self, x0, x1, dtype=tf.float32, use_const=False):
        self.x = tf.placeholder(shape=[x0, x1],dtype=dtype)
        x_data = np.linspace(-1,1, x0* x1).reshape(x0,x1)
        x0 = tf.constant(x_data, dtype=dtype)
        x = tf.multiply(self.x, x0)
        y1 = tf.nn.softmax(x)
        #o0 = tf.add(self.x, y1)
        #o1 = tf.multiply(o0, 0.5)
        self.output = y1

class NetworkMax():
    def __init__(self, x0, x1, dtype=tf.float32, use_const=False):
        self.x = tf.placeholder(shape=[x0, x1],dtype=dtype)
        """
        x_data = np.linspace(-1,1, x0* x1).reshape(x0,x1)
        x0 = tf.constant(x_data, dtype=dtype)
        x = tf.multiply(self.x, x0)"""
        x = self.x
        y1 = tf.reduce_max(x, axis=-1)
        #o0 = tf.add(self.x, y1)
        #o1 = tf.multiply(o0, 0.5)
        self.output = y1

class NetworkTrans():
    def __init__(self, x0, x1, dtype=tf.float32, use_const=False):
        self.x = tf.placeholder(shape=[x0, x1],dtype=dtype)
        x_data = np.linspace(-1,1, x0* x1).reshape(x0,x1)
        x0 = tf.constant(x_data, dtype=dtype)
        x = tf.multiply(self.x, x0)
        y1 = tf.transpose(x)
        #o0 = tf.add(self.x, y1)
        #o1 = tf.multiply(o0, 0.5)
        self.output = y1


class NetworkResnet():
    def __init__(self, x0, x1, x2 , dtype=tf.float32, use_const=False):
        self.x = tf.placeholder(shape=[x0, x1, x1, x2],dtype=dtype)
        off = np.linspace(-1,1, x2).reshape(x2)
        offset = tf.constant(off, dtype=dtype)
        scale = tf.constant(off, dtype=dtype)
        off1 = np.linspace(-0.75, 0.75, x2).reshape(x2)
        offset1 = tf.constant(off1, dtype=dtype)
        scale1 = tf.constant(off1, dtype=dtype)
        x0 = tf.multiply(self.x, scale) # offset=offset, mean=scale, variance=offset, is_training=False)
        x0 = tf.add(x0, offset)
        x1 = tf.nn.relu(x0)
        x2 = tf.multiply(self.x, scale1) # offset=offset, mean=scale, variance=offset, is_training=False)
        x2 = tf.add(x2, offset1)
        y0 = tf.add(x1, x2) 
        #y0 = tf.muliply(y0, scale) # offset=offset, mean=scale, variance=offset, is_training=False)
        #y0 = tf.add(y0, offset)
        #y0 = tf.layers.batch_normalization(x2, training=False)
        y1 = tf.nn.relu(y0)
        #o0 = tf.add(self.x, y1)
        #o1 = tf.multiply(o0, 0.5)
        self.output = y1
        #o1 = y1
        #self.output = tf.add(o1, 0.025)


class Network():
    def __init__(self, x0, x1, y0, y1, use_const=False, dtype=tf.float64, dup = 0):
      
    # x = tf.placeholder(shape=[2048,None],dtype=tf.float32)
    # y = tf.placeholder(shape=[2048,None],dtype=tf.float32)
    # inputW = tf.Variable(tf.random_normal([2048,2048]))
    # inputB = tf.Variable(tf.random_normal([2048,2048]))

    # hideW = tf.Variable(tf.random_normal([2048,2048]))
    # hideB = tf.Variable(tf.random_normal([2048,2048]))

    # h1 = tf.nn.sigmoid(tf.add(tf.matmul(inputW,x),inputB))
    # output = tf.add(tf.matmul(hideW,h1),hideB)
        if not use_const:
            self.x = tf.placeholder(shape=[x0, x1],dtype=dtype)
            self.y = tf.placeholder(shape=[y0, y1],dtype=dtype)
        else:
            x_data = np.linspace(-1,1, x0* x1).reshape(x0,x1)
            y_data = np.linspace(-1,1, y0* y1).reshape(y0,y1)
            self.x = tf.constant(x_data)
            self.y = tf.constant(y_data)

        ta = False
        tb = False    
        if x0 == y0:
            ta = True
        elif x0 == y1:
            ta = True
            tb = True
        elif x1 == y1:
            tb = True
        self.output = tf.matmul(self.x, self.y, transpose_a = ta, transpose_b = tb)

        for _ in range(dup):
            self.output = tf.matmul(self.output, self.y, transpose_a = ta , transpose_b = tb)

    # loss= tf.reduce_mean(tf.reduce_sum(tf.square(y-output)))

    # opt = tf.train.AdamOptimizer(1)

    # train_step = opt.minimize(loss)
def datatype(dt):
    if dt == "float64":
        return tf.float64
    elif dt == "float32":
        return tf.float32



def train_sample(args):
    size = args.size.split(",")
    const = False
    dtype = datatype(args.dtype)
    if args.type == "simple":
        x0 = int(size[0])
        x1 = int(size[1])
        net = NetworkMem(x0, x1, dtype = dtype)
        x_data = np.linspace(-1,1, x0*x1).reshape(x0,x1)
        input_dict = {net.x : x_data} 
    if args.type == "max":
        x0 = int(size[0])
        x1 = int(size[1])
        net = NetworkMax(x0, x1, dtype = dtype)
        x_data = np.linspace(-1,1, x0*x1).reshape(x0,x1)
        input_dict = {net.x : x_data} 
    elif args.type == "trans":
        x0 = int(size[0])
        x1 = int(size[1])
        net = NetworkTrans(x0, x1, dtype = dtype)
        x_data = np.linspace(-1,1, x0*x1).reshape(x0,x1)
        input_dict = {net.x : x_data} 
    elif args.type == "soft":
        x0 = int(size[0])
        x1 = int(size[1])
        net = NetworkSoft(x0, x1, dtype = dtype)
        x_data = np.linspace(-1,1, x0*x1).reshape(x0,x1)
        input_dict = {net.x : x_data} 
    elif args.type == "resnet":
        x0 = int(size[0])
        x1 = int(size[1])
        x2 = int(size[2])
        net = NetworkResnet(x0, x1, x2, dtype = dtype)
        x_data = np.linspace(-1,1, x0*x2*x1*x1).reshape(x0, x1, x1, x2)
        input_dict = {net.x : x_data} 
    elif len(size) == 4: 
        x0 = int(size[0])
        x1 = int(size[1])
        y0 = int(size[2])
        y1 = int(size[3])
        net = Network(x0, x1, y0, y1, const,dtype = dtype,  dup = args.dup)
        x_data = np.linspace(-1,1, x0*x1).reshape(x0,x1)
        y_data = np.linspace(-1,1, y0*y1).reshape(y0,y1)
        input_dict = {net.x : x_data, net.y : y_data} 
    # noise = np.random.normal(0,0.05,x_data.shape)
    # y_data = x_data**3+1+noise
    
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    config.allow_soft_placement = True
    config.log_device_placement = False

    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    train_step = args.times
    if args.timeline:
        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        train_step = 10

    sess.run(init)
    for step in range(args.warm):
        # sess.run(net.train_step,feed_dict={net.x:x_data,net.y:y_data})
        if const:
            pre = sess.run(net.output)
        else:
            pre = sess.run(net.output,feed_dict=input_dict)

    res = []
    st = time.time()

    for step in range(train_step):
        # print('第',step+1,'次训练')
        # sess.run(net.train_step,feed_dict={net.x:x_data,net.y:y_data})
        if args.timeline:
            if const:
                pre = sess.run(net.output, options=run_options, run_metadata=run_metadata)
            else:
                pre = sess.run(net.output,feed_dict=input_dict, options=run_options, run_metadata=run_metadata)
        else:
            if const:
                pre = sess.run(net.output)
            else:
                pre = sess.run(net.output,feed_dict=input_dict)
        #res.append(pre)
    en = time.time()
    print(pre.shape)
    print("time is {}ms per iter {}ms".format((en-st)*1000 , (en-st)*1000 / train_step))
    if args.tb is not None:
        graph = tf.get_default_graph()
        writer = tf.summary.FileWriter(args.tb, graph)
        writer.close()

    if args.timeline:
        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open("tl.json", "w") as f:
            f.write(ctf)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--type", type=str, default="simple")
    parser.add_argument("--tao", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--tb", type=str, default=None)
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--times", type=int, default=100)
    parser.add_argument("--dup", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def check(model_dir="/home/fl237079/checktf/model/"):
    import numpy as np
    from tensorflow.python.saved_model import (tag_constants)
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    config.allow_soft_placement = True
    config.log_device_placement = False
   
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        metagraph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        graph_def = sess.graph_def
        input_dict = {}
        output_dict = []
        input_shape_dict = {}
        output_list = []
        input_list = []
        for signature, ioconfig in metagraph.signature_def.items():
            inputs_mapping = dict(ioconfig.inputs)
            outputs_mapping = dict(ioconfig.outputs)
            for k, v in inputs_mapping.items():
                def get_shape_from_proto(shape_proto):
                    return [dim.size for dim in shape_proto.dim]
                shape = get_shape_from_proto(v.tensor_shape)
                shape = [1 if d <= 0 else d for d in shape]
                input_dict[v.name] = np.random.random(shape).astype(np.float32)
                # if v.dtype == DT_INT64:
                #     input_dict[v.name] = np.random.randint(10, size=shape)
                input_shape_dict[v.name.split(':')[0]] = shape
                input_list.append(v.name.split(':')[0])
            for k, v in outputs_mapping.items():
                output_dict.append(v.name)
                output_list.append(v.name.split(':')[0])

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        metagraph = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)            
        st = time.time() 
        out = sess.run(output_dict, feed_dict=input_dict)
        et = time.time() - st
        out_dict = {}
        i = 0
        for name in output_dict:
            out_dict[name] = out[i]
            i = i+1
        np.save("tf.npy", out_dict)
        print("saved model "  + " warmup time: " + str(et * 1000) + " ms")
        st = time.time()
        for _ in range(1):
            out = sess.run(output_dict, feed_dict=input_dict)
        et = (time.time() - st) * 1000 /  1
        print("saved model "  + " latency: " + str(et) + " ms")




# os.environ["LD_LIBRARY_PATH"] = "{}:{}".format("/home/fl237079/shenshi/env/lib64/python3.6/site-packages/deepmd/op/", os.environ["LD_LIBRARY_PATH"])

# tf.load_op_library("/home/fl237079/shenshi/env/lib64/python3.6/site-packages/deepmd/op/libdeepmd.so")
# tf.load_op_library("/home/fl237079/shenshi/env/lib64/python3.6/site-packages/deepmd/op/libdeepmd_op_rocm.so")
# tf.load_op_library("./libop_abi.so")
# tf.load_op_library("./libop_grads.so")

# @click.command()
# @click.option("--model", type=str)
def load_and_train(model):
  print(model)  
  with tf.Session() as sess:
    meta = '{}.meta'.format(model)
    print(meta)
    saver = tf.train.import_meta_graph(meta)
    saver.restore(sess, "{}".format(model))
    feed_dict = np.load("train_data.npy", allow_pickle=True).item()
    if isinstance(feed_dict, dict):
        print("dict check")
    else:
        print("not dict")
    print(feed_dict)
    trainable_variables = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    # for i in sess.graph_def.node:
    #     print(i.name)
        # apply_op = optimizer.minimize(loss=self.l2_l,
        #                               global_step=self.global_step,
        #                               var_list=trainable_variables,
        #     
        #                           name='train_step')
    ops = tf.get_default_graph().get_operation_by_name("group_deps")
    print(ops)
    globalstep = tf.train.get_or_create_global_step()

    for _ in range(20):
        st = time.time()
        sess.run([ops], feed_dict=feed_dict)
        sess.run(globalstep)
        end = time.time()
        print("{} ms".format((time.time() - st) * 1000))

    
    
    print("***************************123455")
    # check()
    # print("**************************0")

def prepare(args):
    if shutil.which("rocprof") is None:
        root = "/disc"
    else:
        root = "/global/home/aliliang"

    path = root + "/tao_built"
    os.environ["BRIDGE_ENABLE_TAO"] = "true"
    os.environ["TAO_COMPILER_PATH"] = "{}/tao_compiler_main".format(path)
    os.environ["TAO_ENABLE_CHECK"] = "false"
    os.environ["TAO_ENABLE_FALLBACK"] = "true"
    os.environ["TAO_COMPILATION_MODE_ASYNC"] = "false"
    os.environ["TF_XLA_FLAGS"]="--tf_xla_min_cluster_size=0"
    os.environ["TAO_ENABLE_MLIR"]="true"
    os.environ["TAO_MLIR_BRANCH_ONLY"]="true"
    os.environ["TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION"]="true"
    os.environ["TAO_DUMP_PASS_OUTPUT"] = "true"
    import tensorflow as tf
    tf.load_op_library("{}/libtao_ops.so".format(path))
    tf.compat.v1.disable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if args.gpu:
        tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    # print(gpus)




if __name__ == "__main__":
    # check()
    args = parse()
    if args.tao:
        prepare(args)
    # print("**************************0")
    #load_and_train()
    train_sample(args)
    # print("**************************1")
    # check()

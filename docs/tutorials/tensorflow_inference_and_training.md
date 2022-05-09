# Tutorial: Optimize TensorFlow Models with BladeDISC <!-- omit in toc -->

In this tutorial, we show how to optimize TensorFlow models with BladeDISC for
both inference and training. Users only need to add two lines of code to
optimize the model just-in-time for the examples in this tutorial.
Please refer to ["Install BladeDISC With Docker"](/docs/install_with_docker.md)
for environment setup.

The content of this tutorial is as following.
- [BERT CUDA Inference](#bert-cuda-inference)
  - [Prologue: Download Frozen Model](#prologue-download-frozen-model)
  - [All You Need Are the Two Lines!](#all-you-need-are-the-two-lines)
  - [Epilogue: Normal Process to Run Inference](#epilogue-normal-process-to-run-inference)
- [BERT CPU Inference](#bert-cpu-inference)
  - [Prologue: Download Frozen Model For CPU](#prologue-download-frozen-model-for-cpu)
  - [Still, All You Need Are the Two Lines!](#still-all-you-need-are-the-two-lines)
  - [Epilogue: Normal Process to Run Inference on CPU](#epilogue-normal-process-to-run-inference-on-cpu)
- [DeePMD Training](#deepmd-training)
  - [Prologue: Install DeePMD-kit and Download Data](#prologue-install-deepmd-kit-and-download-data)
  - [Aagin, All You Need Are the Two Lines!](#aagin-all-you-need-are-the-two-lines)
  - [Epilogue: Normal Process to Run MD Training with DeePMD-kit API](#epilogue-normal-process-to-run-md-training-with-deepmd-kit-api)


## BERT CUDA Inference

BERT models are usually feed with data of dynamic shapes in real production. The
dynamic shape mainly comes from two aspects, one is varied batch-size, and the
other is varied data shape of each sample (e.g., varied sequence lengths). The
model we show in this tutorial has varied batch-size.

### Prologue: Download Frozen Model

```python
!mkdir -p model
!wget -P model http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bladedisc_notebook_binaries/models/disc_bert_example/frozen.pb
```

### All You Need Are the Two Lines!

All you need to do to optimize the inference is to add the following two lines
of code.
```python
import blade_disc_tf as disc
disc.enable()
```

### Epilogue: Normal Process to Run Inference

After enabling BladeDISC with the two lines above, we can load and run the
frozen model with normal process.

First, we load the frozen model and configure the session.
```python
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

graph_def = tf.GraphDef()
with open('./model/frozen.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
graph = load_frozen_graph()

session_config = tf.ConfigProto()
session_config.allow_soft_placement = True
session_config.gpu_options.allow_growth = True
# Enable auto-mixed-precision in this example.
session_config.graph_options.rewrite_options.auto_mixed_precision = 1
sess = tf.Session(graph = graph, config = session_config)
```

Finally, we prepare input data and run the session. In this example, we fake the
input data with varied batch-size to ease the setup.
```python
for batch in [2, 2, 4, 1, 1, 8, 8, 2, 16, 2]:
    feed_dict = {
        'input_ids_1:0' : np.ones((batch, 384), dtype=int),
        'segment_ids_1:0' : np.zeros((batch, 384), dtype=int),
        'input_mask_1:0' : np.ones((batch, 384), dtype=int),
    }
    outs = sess.run(fetch, feed_dict = feed_dict)
```

The above code shows the complete process to optimize BERT inference model with
BladeDISC. Please refer to [TensorFlow BERT Inference
Example](https://github.com/alibaba/BladeDISC/tree/main/examples/TensorFlow/Inference/CUDA/BERT)
for more scripts to compare the performance of BladeDISC optimization with naive
TensorFlow and XLA.


## BERT CPU Inference

Similar to the CUDA example, The BERT model we show in this tutorial has varied
batch-size.

### Prologue: Download Frozen Model For CPU

```python
!mkdir -p model
!wget -P model http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bladedisc_notebook_binaries/models/disc_bert_cpu_example/frozen.pb
```

### Still, All You Need Are the Two Lines!

Similar to the CUDA example, all you need to optimize the inference is to
add the following two lines of code with some settings that are specific to cpu.
```python
import blade_disc_tf as disc
disc.enable(
  disc_cpu=True, # this enable cpu optimization
  num_intra_op_threads=16 # similar to TF_NUM_INTRAOP_THREADS
)
```

### Epilogue: Normal Process to Run Inference on CPU

After enabling BladeDISC with the two lines above, we can load and run the
frozen model with normal process.

First, we load the frozen model and configure the session.
```python
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

graph_def = tf.GraphDef()
with open('./model/frozen.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
graph = load_frozen_graph()

session_config = tf.ConfigProto()
session_config.inter_op_parallelism_threads = 1
session_config.intra_op_parallelism_threads = 16
sess = tf.Session(graph = graph, config = session_config)
```

Finally, we prepare input data and run the session. In this example, we fake the
input data with varied batch-size to ease the setup.
```python
fetch = ["loss/Softmax:0"]
for batch in [2, 2, 4, 1, 1, 8, 8, 2, 16, 2]:
    feed_dict = {
        'input_ids_1:0' : np.ones((batch, 128), dtype=int),
        'segment_ids_1:0' : np.zeros((batch, 128), dtype=int),
        'input_mask_1:0' : np.ones((batch, 128), dtype=int),
    }
    outs = sess.run(fetch, feed_dict = feed_dict)
```

The above code shows the complete process to optimize BERT inference model with
BladeDISC. Please refer to [TensorFlow BERT Inference
Example on X86](https://github.com/alibaba/BladeDISC/tree/main/examples/TensorFlow/Inference/X86/BERT), or [TensorFlow BERT Inference Example on AArch64](https://github.com/alibaba/BladeDISC/tree/main/examples/TensorFlow/Inference/AArch64/BERT) for more scripts to compare the performance of BladeDISC optimization with naive
TensorFlow and XLA.


## DeePMD Training

We take a deep learning based model for molecular dynamics (MD) to show how to
optimize a training model with BladeDISC. Please refer to
[DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) for more information
about deep learning based MD.



### Prologue: Install DeePMD-kit and Download Data

We need to install DeePMD-kit python interface to run the MD model.
```python
# Install DeePMD-kit according to https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install/install-from-source.md#install-the-python-interface
!git clone https://github.com/deepmodeling/deepmd-kit.git
!cd deepmd-kit; pip install .; cd ..

# Download data
!wget http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bladedisc_notebook_binaries/data/disc_deepmd_example/data.tar.gz
!tar -xzvf data.tar.gz
```

### Aagin, All You Need Are the Two Lines!

All you need to do to optimize the training is to add the following two lines of
code.
```python
import blade_disc_tf as disc
disc.enable()
```

### Epilogue: Normal Process to Run MD Training with DeePMD-kit API

```python
import sys
# The entry point of DeePMD-kit is `main` here.
from deepmd.entrypoints.main import main
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Tell DeePMD-kit that we will perform training with the specified configuration.
sys.argv.append('train')
sys.argv.append('data/input.json')

# Run MD training.
sys.exit(main())
```

The above code shows the complete process to optimize MD training model with
BladeDISC. You can also refer to [TensorFlow DeePMD Training
Example](https://github.com/alibaba/BladeDISC/tree/main/examples/TensorFlow/Train/DeePMD)
for all scripts to optimize MD model.

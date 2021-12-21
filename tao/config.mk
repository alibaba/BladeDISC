# pls set according to your envrionment setting, e.g.:
# TF_CFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# TF_LFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

TF_CFLAGS := -I/usr/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DTF_1_12
TF_LFLAGS := -L/usr/lib/python2.7/site-packages/tensorflow -ltensorflow_framework

# Note that it should be the same version with the Tensorflow used.
PROTO := /state/dev/xla/features/control_flow/tensorflow/bazel-out/host/bin/external/protobuf_archive/protoc

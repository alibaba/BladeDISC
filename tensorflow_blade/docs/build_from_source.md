# Build Tensorflow-Blade from Source

This document introduces how to build Tensorflow-Blade from source.
As for the preparation part, you can refer to **Prerequisite**, **Checkout the Source**, **Launch a development Docker container**
parts from [build from sources docs for BladeDisc for TensorFlow Users](../../docs/build_from_source.md).

## Building Tensorflow-Blade

Build/test/package for Tensorflow-Blade are all done by bazel.
We have provided an all-in-on script for developers/users to use instead of using complex bazel command.

### options for build script
``` bash
#./build.py -h
usage: build.py [-h] [-s stage] [--device {cpu,gpu}] [--tf {1.15,2.4}]
                [--skip-trt] [--skip-hie] [--internal] [--verbose] [--develop]
                python_dir

positional arguments:
  python_dir
                                    Directory of virtualenv where target tensorflow installed.
                                    If not specificed, the default python will be used(from `which python3`).

optional arguments:
  -h, --help            show this help message and exit
  -s stage, --stage stage
                        Run all or a single build stage or sub-stage, it can be:
                            - all: The default, it will configure, build, test and package.
                            - check: Run format checkers and static linters.
                            - configure: parent stage of the following:
                            - build: build tf blade with regard to the configured part
                            - test: test tf blade framework.
                            - package: make tf blade python packages.
  --device {cpu,gpu}    Build target device
  --tf {1.15,2.4}       Tensorflow version.
  --skip-trt            If True, tensorrt will be skipped for gpu build
  --skip-hie            If True, hie will be skipped for internal build
  --internal            If True, internal objects will be built
  --verbose             Show more information in each stage
  --develop             If True, python wheel develop mode will be installed for local development or debug.
```

### A typical build flow for GPU device and Tensorflow-gpu==2.4.0
 - 1. Create an venv for build
```bash
# create a simple venv in docker
python -m virtualenv ${VENV_PATH} && source ${VENV_PATH}/bin/activate
python -m pip install -r requirement-tf2.4-cu110.txt
```
 - 2. Configure for bazel build
```bash
./build.py ${VENV_PATH} -s configure
```
After configuration, all the generated options can be found in .bazelrc_gen file in tensorflow_blade dir.
```bash
head .bazelrc_gen

build --cxxopt=-std=c++14
build --host_cxxopt=-std=c++14
build --compilation_mode=opt
build --cxxopt=-DBUILD_WITH_BAZEL
build --action_env BUILD_WITH_BAZEL=1
build --action_env BLADE_WITH_TF=1
build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0
build --host_cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0
```
 - 3. build or package
**-s build** command wile build all the cpp related targets -- **_tf_blade.so and libtf_blade.so**
```bash
./build.py ${VENV_PATH} -s build
```

**-s package** command will generate python wheel package for tensorflow-blade under the dir of tensorflow_blade/dist
The wheel should have the name similar to **tensorflow_blade_gpu-0.0.0+cu110-py3-none-any.whl**
```bash
./build.py ${VENV_PATH} -s package
```

If you want to install the wheel package under develop mode for debug, the following command can be applied.
```bash
./build.py ${VENV_PATH} -s package --develop
```

## NOTE
Currently only tensorflow-gpu==2.4.0 with CUDA 11.0 is supported for Tensorflow-Blade.

Supports for more tensorflow versions and devices will be coming soon.

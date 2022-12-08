# TensorFlow-Blade
Blade is an optimization toolkit that support multiple tools (aka, **TensorRT** and **DISC** for open sourced version) as backend. It facilitates users in using TensorRT as backend transparently, and provides automatic selection between TensorRT and DISC provided with specific workloads. The automatic selection logics will be opensourced soon. TensorFlow-Blade is Blade for TensorFlow frontends.

## build and use TensorFlow-Blade
**NOTE: Currently only `python3` is supported.**

[build and install TensorFlow-Blade](docs/build_from_source.md)

[optimize BERT model with TensorFlow-Blade](docs/tutorials/tensorflow_blade_bert_inference.md)

## design of tf2trt optimization(WIP)

## Make contribution for TensorFlow-Blade
First ref to [How to make contribution](/docs/contribution.md) to setup the basic develop environment and fork code to your own branch.
Then we will get to know the code for TensorFlow-Blade.

### Code layout for TensorFlow-Blade
All the code for TensorFlow-Blade are under the `tensorflor_blade` dir.

```bash
tree -L 1
.
├── BUILD
├── build_defs.bzl
├── build_pip_package.sh
├── build.py
├── common_internal.py
├── dist
├── docs
├── examples
├── README.md
├── requirement-tf2.4-cu110.txt
├── setup.py
├── src
├── tests
├── tf_blade
├── venv
├── version.py
├── WORKSPACE
├── workspace0.bzl
├── workspace1.bzl
└── workspace2.bzl
```
All the files with name containing `workspace` and `build` are all used for **bazel build**.
All the cpp source codes are under `src` dir.
All the python source codes are under `tf_blade` dir.
All the tests files are under `tests` dir.

### CI pipeline for TensorFlow-Blade
When running CI actions for TensorFlow-Blade, all the `configure/build/package` stages as in [build and install TensorFlow-Blade](docs/build_from_source.md) are executed. Also we will run the python linter check and run all the tests under tests dir.
 - run checks for python and cpp code
```bash
./build.py -s check
```
The checks will be done using python's `flake8/black/mypy` modules.
Please make sure all the python functions or classes have `type hints` as we have done in current code.

As for cpp code, the check will be done by the `pre-commit` tool using clang-format.

 - run unit tests
```bash
./build.py -s test
```
All the unit tests under `tests/` dir will be executed.

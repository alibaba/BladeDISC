#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source ${SCRIPT_DIR}/parse_args.sh "$@"

ENTRY=scripts/python/tao_build.py
VENV_PATH=/opt/venv_disc
BLADE_DISC_DIR=tao/python/blade_disc_tf

# cleanup build cache
(rm -rf build \
  && rm -rf tao/build \
  && cd tf_community && bazel clean --expunge)

python ${ENTRY} ${VENV_PATH} -s configure --bridge-gcc default --compiler-gcc default ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s build_tao_bridge ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s build_tao_compiler ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s build_mlir_ral ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s test_tao_bridge_cpp ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s test_tao_bridge_py ${CPU_ONLY}
python ${ENTRY} ${VENV_PATH} -s test_tao_compiler ${CPU_ONLY}

# copy libtao_ops.so and tao_compiler_main to blade-disc-tf
cp tao/build/libtao_ops.so ${BLADE_DISC_DIR}
cp tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main ${BLADE_DISC_DIR}

# test and build blade-disc-tf Python package
(cd tao && \
  ${VENV_PATH}/bin/pytest --pyargs python
  ${VENV_PATH}/bin/python setup.py bdist_wheel)

# copy Python wheel package to build folder
mkdir -p build && \
cp tao/dist/blade_disc*.whl ./build

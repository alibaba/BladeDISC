#!/usr/bin/env python3
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



# E501: max line length checking
# flake8: noqa: E501

from __future__ import print_function

import argparse
import glob
import os
import re
import shutil
import socket
import tarfile
import time
import fnmatch
from datetime import datetime

from common_setup import (
    StageTiming,
    stage_time,
    time_stage,
    build_mkldnn,
    config_mkldnn,
    mkl_install_dir,
    symlink_files,
    ensure_empty_dir,
    cwd,
    get_source_root_dir,
    logger,
    which,
    running_on_ci,
    ci_build_flag,
    remote_cache_token,
    update_cpu_specific_setting,
    acl_root_dir,
    get_tf_info,
    deduce_cuda_info,
)

from tao_common import (
    VALID_GCC,
    git_branch,
    git_head,
    get_tf_gpu_version,
    execute,
    default_env,
    gcc_env,
    overwrite_file,
)

PYTHON_BIN_NAME = os.getenv("PYTHON", "python")

def get_version_file(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "VERSION")


def tf_root_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tf_community")


def tao_compiler_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao_compiler")


def tao_bridge_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "tao_bridge")


def tao_build_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "build")

def tao_bazel_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao")

def tao_ral_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "tao_bridge", "ral")

def internal_root_dir():
    return os.path.join(get_source_root_dir(), os.pardir)

def internal_tao_bridge_dir():
    return os.path.join(internal_root_dir(), "platform_alibaba", "tao_bridge")

def link_internal_tao_bridge(args):
    # softlink ["tao_launch_op", "gpu"] dirs, "tvm" and "transform" dirs are not needed for now.
    for dir_name in ["tao_launch_op", "gpu"]:
        src_file = os.path.join(internal_tao_bridge_dir(), dir_name)
        link_in_bridge = os.path.join(tao_bridge_dir(), dir_name)
        if args.platform_alibaba:
            execute("rm -rf {0} && ln -s {1} {0}".format(link_in_bridge, src_file))
        else:
            execute("rm -rf {0}".format(link_in_bridge))

def add_ral_link_if_not_exist(root):
    RAL_DIR_IN_TF = "tao_compiler/mlir/xla"
    PROTO = "compile_metadata.proto"
    RAL_DIR_IN_BRIDGE = os.path.join(tao_ral_dir(root), "tensorflow/compiler/mlir/xla")
    if os.path.exists(RAL_DIR_IN_BRIDGE):
        shutil.rmtree(RAL_DIR_IN_BRIDGE)
    os.makedirs(RAL_DIR_IN_BRIDGE)
    with cwd(RAL_DIR_IN_BRIDGE):
        execute("ln -s {0}/{1}/ral ral".format(root, RAL_DIR_IN_TF))
        execute("ln -s {0}/{1}/ral/{2} {2}".format(root, RAL_DIR_IN_TF, PROTO))


def tao_ci_conf_file():
    root = get_source_root_dir()
    return os.path.join(root, "scripts", "ci", ".tao_ci_conf")


def save_gcc_conf(args):
    """Save gcc conf to ${ROOT}/scripts/ci/.tao_ci_conf"""
    with open(tao_ci_conf_file(), "w") as f:
        if args.bridge_gcc is not None:
            f.write("bridge_gcc=" + args.bridge_gcc + "\n")
        if args.compiler_gcc is not None:
            f.write("compiler_gcc=" + args.compiler_gcc + "\n")


def restore_gcc_conf(args):
    """Save gcc conf to ${ROOT}/scripts/ci/.tao_ci_conf"""
    with open(tao_ci_conf_file(), "r") as f:
        for line in f.readlines():
            k, v = line.rstrip().split("=")
            k = k.strip()
            v = v.strip()
            setattr(args, k, v)

def symlink_internal_files(root):
    with cwd(root):
        logger.info("linking PatineClient")
        execute("rm -rf {0} && ln -s {1} {0}".format(os.path.join('tf_community', 'tao', 'third_party', 'PatineClient'),
                os.path.join(internal_root_dir(), 'platform_alibaba', 'third_party', 'PatineClient')))
        logger.info("linking blade_gemm")
        execute("rm -rf {0} && ln -s {1} {0}".format(os.path.join('tf_community', 'tao', 'blade_gemm'),
                os.path.join(internal_root_dir(), 'platform_alibaba', 'blade_gemm')))

        logger.info("cleanup tao_compiler with XLA always...")
        src = os.path.join(internal_root_dir(), "platform_alibaba/tao_compiler/xla")
        dest = "tf_community/tensorflow/compiler/decoupling_xla"
        execute("rm -rf {1} && ln -s {0} {1}".format(src, dest))


def configure_compiler(root, args):
    symlink_files(root)
    if args.platform_alibaba:
        symlink_internal_files(root)

    # configure tensorflow
    with cwd(tf_root_dir()), gcc_env(args.compiler_gcc):
        cmd = "set -a && source {} && set +a &&".format(
            os.getenv("TAO_COMPILER_BUILD_ENV")
        )
        cmd += " env USE_DEFAULT_PYTHON_LIB_PATH=1"
        # tf master doesn't support python2 any more
        cmd += " PYTHON_BIN_PATH=" + which("python3")
        cmd += ' CC_OPT_FLAGS="-Wno-sign-compare"'
        cmd += " ./configure"
        logger.info(cmd)
        execute(cmd)

        # TF_REMOTE_CACHE is not supported by tensorflow community
        # just set remote cache here
        token = remote_cache_token()
        if token:
            with open(".tf_configure.bazelrc", "a") as f:
                f.write("\n")
                f.write("build --remote_cache={}\n".format(token))
                f.write("test --remote_cache={}\n".format(token))
        with open(".tf_configure.bazelrc", "a") as f:
            f.write("\n")
            bazel_startup_opts = "--host_jvm_args=-Djdk.http.auth.tunneling.disabledSchemes="
            f.write("startup {}\n".format(bazel_startup_opts))

    logger.info("Stage [configure compiler] success.")

@time_stage()
def configure_pytorch(root, args):
    save_gcc_conf(args)
    logger.info("configuring aicompiler for pytorch ......")
    configure_compiler(root, args)

@time_stage()
def configure_bridge_cmake(root, args):
    save_gcc_conf(args)
    add_ral_link_if_not_exist(root)
    tao_bridge_build_dir = tao_build_dir(root)
    ensure_empty_dir(tao_bridge_build_dir, clear_hidden=False)
    with cwd(tao_bridge_build_dir), gcc_env(args.bridge_gcc):
        cc = which("gcc")
        cxx = which("g++")
        envs = " CC={} CXX={} ".format(cc, cxx)
        if args.build_in_tf_addon:
            # called from tensorflow_addons
            flags = "-DTAO_ENABLE_WARMUP_XFLOW=OFF"
        else:
            # default flags
            flags = "-DTAO_ENABLE_CXX_TESTING=ON"
        flags += " -DTAO_DISABLE_LINK_TF_FRAMEWORK={} ".format(
            "ON" if args.disable_link_tf_framework else "OFF"
        )
        if args.enable_blaze_opt:
            flags += " -DBLAZE_OPT=true"
        flags += " -DTAO_CPU_ONLY={}".format(args.cpu_only)
        flags += " -DTAO_DCU={}".format(args.dcu)
        is_cuda = not (args.cpu_only or args.dcu)
        flags += " -DTAO_CUDA={}".format(is_cuda)
        flags += " -DTAO_ENABLE_MKLDNN={} ".format(
            "ON" if args.enable_mkldnn else "OFF"
        )
        if args.enable_mkldnn:
            flags +=" -DMKL_ROOT={} ".format(mkl_install_dir(root))
        flags += " -DTAO_X86={}".format(args.x86)
        flags += " -DTAO_AARCH64={}".format(args.aarch64)
        if args.aarch64:
            acl_root = acl_root_dir(root)
            envs += " ACL_ROOT_DIR={} ".format(acl_root)
            flags += " -DDNNL_AARCH64_USE_ACL=ON "

        cmake_cmd = (
            "{} cmake .. -DPYTHON={}/bin/{} {}".format(
                envs, args.venv_dir, PYTHON_BIN_NAME, flags
            )
        )
        logger.info("configuring tao_bridge ......")
        execute(cmake_cmd)

    with cwd(root):
        # copy version.h from tao_bridge
        execute(
            "cp {}/tao_bridge/version.h tao_compiler/decoupling/version.h".format(
                tao_bridge_build_dir
            )
        )
    logger.info("Stage [configure bridge(cmake)] success.")

@time_stage()
def configure_bridge_bazel(root, args):
    save_gcc_conf(args)
    # TODO(lanbo.llb): support tf_addons build with bazel
    # TODO(lanbo.llb): support TAO_DISABLE_LINK_TF_FRAMEWORK in bazel??
    tao_bazel_root = tao_bazel_dir(root)
    link_internal_tao_bridge(args)
    with open(os.path.join(tao_bazel_root, ".bazelrc_gen"), "w") as f:

        def _opt(opt, value, cmd="build"):
            f.write(f"{cmd} --{opt}={value}\n")

        def _action_env(key, value, cmd="build"):
            f.write(f"{cmd} --action_env {key}={value}\n")

        def _write(line, cmd="build"):
            f.write(f"{cmd} {line}\n")

        python_bin = os.path.join(args.venv_dir, "bin", "python3")
        _action_env("PYTHON_BIN_PATH", python_bin)
        _action_env("GCC_HOST_COMPILER_PATH", which("gcc"))
        _action_env("CC", which("gcc"))
        _action_env("CXX", which("g++"))
        (
            tf_major,
            tf_minor,
            is_pai,
            tf_header_dir,
            tf_lib_dir,
            tf_lib_name,
            tf_cxx11_abi,
            tf_pb_version,
        ) = get_tf_info(python_bin)
        _opt("cxxopt", f"-D_GLIBCXX_USE_CXX11_ABI={tf_cxx11_abi}")
        _opt("host_cxxopt", f"-D_GLIBCXX_USE_CXX11_ABI={tf_cxx11_abi}")
        _action_env("BLADE_WITH_TF", "1")
        _action_env("IF_CXX11_ABI", int(tf_cxx11_abi))
        _action_env("TF_IS_PAI", int(is_pai))
        _action_env("TF_MAJOR_VERSION", tf_major)
        _action_env("TF_MINOR_VERSION", tf_minor)
        _action_env("TF_HEADER_DIR", tf_header_dir)
        _action_env("TF_SHARED_LIBRARY_DIR", tf_lib_dir)
        _action_env("TF_SHARED_LIBRARY_NAME", tf_lib_name)
        _action_env("TF_PROTOBUF_VERSION", tf_pb_version)
        # Build environments. They all starts with `DISC_BUILD_`.
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        _action_env("DISC_BUILD_VERSION", args.version)
        _action_env("DISC_BUILD_GIT_BRANCH", git_branch().decode("utf-8").replace('/', '-'))
        _action_env("DISC_BUILD_GIT_HEAD", git_head().decode("utf-8"))
        _action_env("DISC_BUILD_HOST", host)
        _action_env("DISC_BUILD_IP", ip)
        _action_env("DISC_BUILD_TIME", datetime.today().strftime("%Y%m%d%H%M%S"))

        is_cuda = not (args.cpu_only or args.dcu)
        if is_cuda:
            cuda_ver, _ = deduce_cuda_info()
            if '11\.' in cuda_ver:
                _action_env("TF_CUDA_COMPUTE_CAPABILITIES", "7.0,7.5,8.0")
            elif '10\.' in cuda_ver:
                _action_env("TF_CUDA_COMPUTE_CAPABILITIES", "7.0,7.5")
            _action_env("NVCC", which("nvcc"))
            _write("--test_tag_filters=-cpu", cmd="test")
        elif args.cpu_only:
            _write("--test_tag_filters=-gpu", cmd="test")
            if args.enable_mkldnn:
                _action_env("BUILD_WITH_MKLDNN", "1")
            if args.aarch64:
                _action_env("BUILD_WITH_AARCH64", "1")

        _write("--host_jvm_args=-Djdk.http.auth.tunneling.disabledSchemes=", cmd = "startup")
        logger.info("configuring tao_bridge with bazel ......")

    with cwd(tao_bazel_root), gcc_env(args.bridge_gcc):
        # make sure version.h is generated
        execute("bazel build --config=release //:version_header_genrule")

    with cwd(root):
        # copy version.h from tao_bridge
        # NOTE(lanbo.llb): This is no longer needed when tao_compiler is build
        # in workspace `org_tao_compiler` instead of `org_tensorflow`
        execute(
            f"cp {tao_bazel_root}/bazel-bin/tao_bridge/version.h tao_compiler/decoupling/version.h"
        )
    logger.info("Stage [configure bridge(bazel)] success.")


@time_stage()
def configure(root, args):
    if args.cmake:
        configure_bridge_cmake(root, args)
    else:
        configure_bridge_bazel(root, args)
    configure_compiler(root, args)


@time_stage()
def build_tao_compiler(root, args):
    BAZEL_BUILD_CMD = "bazel build --experimental_multi_threaded_digest --define framework_shared_object=false" + ci_build_flag()
    TARGET_TAO_COMPILER_MAIN = "//tensorflow/compiler/decoupling:tao_compiler_main"
    TARGET_DISC_OPT = "//tensorflow/compiler/mlir/disc:disc-opt"
    TARGET_DISC_REPLAY = "//tensorflow/compiler/mlir/disc/tools/disc-replay:disc-replay-main"

    targets = None
    if args.bazel_target is not None:
        targets = set(args.bazel_target.split(","))

    @time_stage(incl_args=[0])
    def bazel_build(target, flag=""):
        if targets is not None and target not in targets:
            logger.info("Skip bazel target: " + target)
            return
        logger.info("Building bazel target: " + target)
        execute(" ".join([BAZEL_BUILD_CMD, flag, target]))

    with cwd(tf_root_dir(root)), gcc_env(args.compiler_gcc):
        execute(
            "cp -f -p {}/tao*.proto tensorflow/compiler/decoupling/".format(
                tao_bridge_dir(root)
            )
        )

        if args.cpu_only:
            if args.aarch64:
                flag = '--config=disc_aarch64 '
            else:
                flag = '--config=disc_x86 '
        elif args.dcu:
            flag = "--config=dcu"
        else:
            flag = "--config=cuda"
            if args.platform_alibaba and args.blade_gemm:
                flag += " --config=blade_gemm"

        if args.platform_alibaba:
            flag += " --config=platform_alibaba"

        if args.build_dbg_symbol:
            flag += " --copt=-g"

        if args.enable_blaze_opt:
            flag += ' --cxxopt="-DBLAZE_OPT"'

        if args.enable_mkldnn:
            flag += ' --config=disc_mkldnn'

        bazel_build(TARGET_TAO_COMPILER_MAIN, flag=flag)
        bazel_build(TARGET_DISC_OPT, flag=flag)
        bazel_build(TARGET_DISC_REPLAY, flag=flag)
        execute(
            "cp -f -p {}/tao/third_party/ptxas/10.2/ptxas ./bazel-bin/tensorflow/compiler/decoupling/".format(
                root
            )
        )
    logger.info("Stage [build_tao_compiler] success.")


@time_stage()
def build_mlir_ral(root, args):
    configs = ['--config=cxx11abi_{}'.format(int(args.ral_cxx11_abi))]
    if not args.cpu_only:
        if args.dcu:
            configs.append('--config=disc_dcu')
        else:
            configs.append('--config=disc_cuda')
            if args.platform_alibaba and args.blade_gemm:
                configs.append('--config=blade_gemm')
    else:
        if args.aarch64:
            configs.append('--config=disc_aarch64')
        else:
            configs.append('--config=disc_x86')

    if args.platform_alibaba:
        configs.append(" --config=platform_alibaba")



    if args.enable_blaze_opt:
        configs.append('--config=disc_blaze')

    if args.enable_mkldnn:
        configs.append('--config=disc_mkldnn')

    if running_on_ci():
        configs.append('--config=ci_build')

    BAZEL_BUILD_CMD = "bazel build --config=disc "
    BAZEL_BUILD_CMD = BAZEL_BUILD_CMD + " ".join(configs)

    TARGET_RAL_STANDALONE_LIB = "//tensorflow/compiler/mlir/xla/ral:libral_base_context.so"
    TARGET_DHLO_COMPILER_MAIN = "//tensorflow/compiler/mlir/disc:disc_compiler_main"
    TARGET_MLIR_DISC_BUILDER = "//tensorflow/compiler/mlir/disc:mlir_disc_builder.so"
    TARGET_MLIR_DISC_BUILDER_HEADER = "//tensorflow/compiler/mlir/disc:install_mlir_disc_headers"

    def bazel_build(target, flag=""):
        logger.info("Building bazel target: " + target)
        execute(" ".join([BAZEL_BUILD_CMD, flag, target]))

    flag = ""
    with cwd(tf_root_dir(root)), gcc_env(args.bridge_gcc):
        bazel_build(TARGET_RAL_STANDALONE_LIB, flag=flag)
        bazel_build(TARGET_MLIR_DISC_BUILDER, flag=flag)
        bazel_build(TARGET_MLIR_DISC_BUILDER_HEADER, flag=flag)

    with cwd(tf_root_dir(root)), gcc_env(args.compiler_gcc):
        if not args.cpu_only:
            # A workaround for a bug of gcc<=7.3 since devtoolset-7 supports up to 7.3.1
            # and cuda-10 runtime cannot support devtools-8 for now.
            # Revisit this if upgrade to devtools-8.
            # Refer to: https://github.com/tensorflow/tensorflow/issues/25323
            execute("sed -i \
                's/values\[i\] = coeff(index+i);/Self::CoeffReturnType t = coeff(index+i);values\[i\] = t;/g' \
                'bazel-tf_community/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor/TensorImagePatch.h'")
            execute("sed -i \
                '/values\[i\] = internal::InnerMostDimReducer<Self, Op>::reduce(\*this, firstIndex + i \* num_values_to_reduce,$/{n;d}' \
                'bazel-tf_community/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h'")
            execute("sed -i \
                's/values\[i\] = internal::InnerMostDimReducer<Self, Op>::reduce(\*this, firstIndex + i \* num_values_to_reduce,$/CoeffReturnType t = internal::InnerMostDimReducer<Self, Op>::reduce(\*this, firstIndex + i \* num_values_to_reduce, num_values_to_reduce, reducer)\;values\[i\] = t\;/g' \
                'bazel-tf_community/external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h'")
        bazel_build(TARGET_DHLO_COMPILER_MAIN, flag=flag)

    logger.info("Stage [build_mlir_ral] success.")

@time_stage()
def test_tao_compiler(root, args):
    BAZEL_BUILD_CMD = "bazel build --experimental_multi_threaded_digest --define framework_shared_object=false --test_timeout=600 --javabase=@bazel_tools//tools/jdk:remote_jdk11"
    BAZEL_TEST_CMD = "bazel test --experimental_multi_threaded_digest --define framework_shared_object=false --test_timeout=600 --javabase=@bazel_tools//tools/jdk:remote_jdk11"
    BAZEL_TEST_CMD += ci_build_flag()
    BAZEL_BUILD_CMD += ci_build_flag()
    if running_on_ci():
        # NOTE: using the lower parallel jobs on CI host to avoid OOM
        BAZEL_TEST_CMD += " --jobs=10"
    else:
        BAZEL_TEST_CMD += " --jobs=30"

    TARGET_DISC_TRANSFORMS_TEST = "//tensorflow/compiler/mlir/disc/transforms/tests/..."
    TARGET_DISC_E2E_TEST = "//tensorflow/compiler/mlir/disc/tests/..."

    TARGET_DISC_REPLAY_TEST = "//tensorflow/compiler/mlir/disc/tools/disc-replay:disc-replay-test"

    targets = None
    if args.bazel_target is not None:
        targets = set(args.bazel_target.split(","))

    @time_stage(incl_args=[0])
    def bazel_test(target, flag=""):
        if targets is not None and target not in targets:
            return
        logger.info("Testing bazel target: " + target)
        execute(" ".join([BAZEL_BUILD_CMD, flag, target]))
        execute(" ".join([BAZEL_TEST_CMD, flag + ' --test_env=TF_CPP_VMODULE=disc_compiler=1' , target]))

    with cwd(tf_root_dir(root)), gcc_env(args.compiler_gcc):
        execute(
            "cp -f -p {}/tao*.proto tensorflow/compiler/decoupling/".format(
                tao_bridge_dir(root)
            )
        )
        if args.cpu_only:
            if args.aarch64:
                flag = '--config=disc_aarch64 '
            else:
                flag = '--config=disc_x86 '
            if args.enable_mkldnn:
                flag += ' --config=disc_mkldnn'
            if args.platform_alibaba:
                flag += " --config=platform_alibaba"
            mlir_test_list = [
                TARGET_DISC_TRANSFORMS_TEST,
                TARGET_DISC_E2E_TEST,
            ]
            MLIR_TESTS = " ".join(mlir_test_list)
            bazel_test(MLIR_TESTS, flag=flag)
        else:
            if args.dcu:
                flag = "--config=dcu"
            else:
                flag = "--config=cuda"
                if args.platform_alibaba and args.blade_gemm:
                    flag += ' --config=blade_gemm'
            if args.platform_alibaba:
                flag += " --config=platform_alibaba"
            mlir_tests_list = [
                TARGET_DISC_TRANSFORMS_TEST,
                TARGET_DISC_E2E_TEST,
                TARGET_DISC_REPLAY_TEST
            ]
            MLIR_TESTS = " ".join(mlir_tests_list)
            bazel_test(MLIR_TESTS, flag=flag)
            flag += " --action_env=BRIDGE_ENABLE_TAO=true "
    logger.info("Stage [test_tao_compiler] success.")

def tao_bridge_bazel_config(args):
    bazel_config = ""
    if args.enable_blaze_opt:
        bazel_config += " --config=disc_blaze"
    if args.cpu_only:
        if args.x86:
            bazel_config += " --config=disc_x86"
        elif args.aarch64:
            bazel_config += " --config=disc_aarch64"
        if args.enable_mkldnn:
            bazel_config += " --config=disc_mkldnn"
    elif args.dcu:
        bazel_config += " --config=disc_dcu"
    else:
        bazel_config += " --config=disc_cuda"
        if args.platform_alibaba and args.blade_gemm:
            bazel_config += " --config=blade_gemm"
    if args.platform_alibaba:
        bazel_config += " --config=platform_alibaba"
    return bazel_config

@time_stage()
def build_tao_bridge(root, args):
    if args.cmake:
        tao_bridge_build_dir = tao_build_dir(root)
        with cwd(tao_bridge_build_dir), gcc_env(args.bridge_gcc):
            execute("make -j")
    else:
        tao_bazel_root = tao_bazel_dir(root)
        with cwd(tao_bazel_root), gcc_env(args.bridge_gcc):
            execute(f"bazel build {tao_bridge_bazel_config(args)} //:libtao_ops.so")

    logger.info("Stage [build_tao_bridge] success.")


@time_stage()
def build_dsw(root, args):
    dsw_build_dir = os.path.join(root, "platform_alibaba", "tools", "tao")
    # copy VERSION file
    overwrite_file(get_version_file(), os.path.join(dsw_build_dir, "tao", "VERSION"))

    with cwd(dsw_build_dir), gcc_env(args.bridge_gcc):
        # remove previous build results
        for tmpdir in ["build", "dist", "tao.egg-info"]:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        execute("{}/bin/python setup.py bdist_wheel --universal".format(args.venv_dir))
    logger.info("Stage [build_dsw] success.")

def py_test_reader(root, includes):
    includes = r'|'.join([fnmatch.translate(x) for x in includes])
    for dirpath, _, files in os.walk(root):
        files = [f for f in files if re.match(includes, f)]
        files = [os.path.join(dirpath, f) for f in files]
        for fname in files:
            yield fname

# it seems pytest or forked plugin has some strange bug
# sometimes we can run each test*.py successfully separately
# but it failed when we kick off the run in one command
def run_py_test(py_bin, test_root, output_file, includes, envs=[]):
    failed_list = list()
    env_str = " ".join(envs)
    for test in py_test_reader(test_root, includes):
        try:
            execute(
                "{} {} -m pytest --forked {} 2>&1 | tee -a {}".format(
                    env_str, py_bin, test, output_file
                )
            )
        except Exception as e:
            failed_list.append((test, e))

    if len(failed_list) > 0:
        logger.error("=== Total {} python tests failed ===".format(len(failed_list)))
        for i, v in enumerate(failed_list):
            logger.error("{}: {}\n{}".format(i, v[0], v[1]))
        raise Exception("pytest failed: " + test_root)
    else:
        logger.info("pytest passed: " + test_root)


@time_stage()
def test_tao_bridge(root, args, cpp=True, python=True):
    if args.cmake:
        tao_bridge_build_dir = tao_build_dir(root)
        # Perform test within tao bridge GCC environment.
        with cwd(tao_bridge_build_dir), gcc_env(args.bridge_gcc):
            if cpp:
                output_file = os.path.join(tao_bridge_build_dir, "cpp_test.out")
                execute("make test ARGS='-V' | tee {}".format(output_file))
                logger.info("Stage [test_tao_bridge_cpp] success, output: " + output_file)
            if python:
                output_file = os.path.join(tao_bridge_build_dir, "py_test.out")
                py_bin = os.path.join(args.venv_dir, "bin", PYTHON_BIN_NAME)
                test_root_disc = "{}/tao/tao_bridge/test/gpu".format(root)
                if not args.cpu_only:
                    run_py_test(py_bin, test_root_disc, output_file, ["test_mlir*.py"])

                logger.info("Stage [test_tao_bridge_py] success, output: " + output_file)
    else:
        tao_bazel_root = tao_bazel_dir(root)
        with cwd(tao_bazel_root), gcc_env(args.bridge_gcc):
            if cpp:
                output_file = os.path.join(tao_bazel_root, "cpp_test.out")
                execute(f"bazel test {tao_bridge_bazel_config(args)} //...")
                logger.info("Stage [test_tao_bridge_cpp] with bazel success, output: " + output_file)
            if python:
                output_file = os.path.join(tao_bazel_root, "py_test.out")
                py_bin = os.path.join(args.venv_dir, "bin", PYTHON_BIN_NAME)
                test_root_disc = "{}/tao/tao_bridge/test/gpu".format(root)
                test_root_cpu = "{}/platform_alibaba/tao_bridge/test/cpu".format(internal_root_dir())
                test_root_gpu = "{}/platform_alibaba/tao_bridge/test/gpu".format(internal_root_dir())

                if args.platform_alibaba:
                    if args.cpu_only:
                        run_py_test(py_bin, test_root_cpu, output_file, ["test_*.py"],
                            ["PLATFORM_ALIBABA=ON"])
                    else:
                        run_py_test(py_bin, test_root_gpu, output_file, ["test_*.py"],
                            ["PLATFORM_ALIBABA=ON"])
                        run_py_test(py_bin, test_root_disc, output_file, ["test_mlir*.py"],
                            ["PLATFORM_ALIBABA=ON"])
                else:
                    if not args.cpu_only:
                        run_py_test(py_bin, test_root_disc, output_file, ["test_mlir*.py"])

                logger.info("Stage [test_tao_bridge_py] success, output: " + output_file)


def prepare_env(args):
    """Prepare env vars related to building ."""
    # Build configurations.
    os.environ["TF_PLATFORM"] = "tao"
    if args.dcu:
        os.environ["TF_DEVICE"] = "dcu"
        logger.info("[BUILD] build for DCU ...")
    elif not args.cpu_only:
        os.environ["TF_DEVICE"] = "gpu"
        os.environ["TF_GPU_VERSION"] = get_tf_gpu_version()
        logger.info("[BUILD] TF_GPU_VERSION: " + os.environ["TF_GPU_VERSION"])
    else:
        os.environ["TF_DEVICE"] = "cpu"
    if "TF_BUILD_ENABLE_DEBUG_SYMBOLS" not in os.environ:
        os.environ["TF_BUILD_ENABLE_DEBUG_SYMBOLS"] = "0"

    # load env config
    config_file = os.path.join(
        get_source_root_dir(),
        "tao_compiler",
        "ci_build",
        "platforms",
        os.getenv("TF_PLATFORM"),
        os.getenv("TF_DEVICE"),
        "env.conf",
    )
    if os.getenv("TF_DEVICE") == "gpu":
        config_file += "." + os.getenv("TF_GPU_VERSION")
    assert os.path.exists(config_file), "{} not exists".format(config_file)
    os.environ["TAO_COMPILER_BUILD_ENV"] = config_file

    # Build environments. They all starts with `TAO_BUILD_`.
    os.environ["TAO_BUILD_VERSION"] = args.version
    os.environ["TAO_BUILD_GIT_BRANCH"] = str(git_branch())
    os.environ["TAO_BUILD_GIT_HEAD"] = str(git_head())
    host = socket.gethostname()
    os.environ["TAO_BUILD_HOST"] = host
    os.environ["TAO_BUILD_IP"] = socket.gethostbyname(host)
    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    os.environ["TAO_BUILD_TIME"] = timestamp

def generate_build_info(file):
    with open(file, "w") as f:
        f.write(
            "# Use the following command to check version info embedded in libtao_ops.so:\n"
        )
        f.write(
            "#     {} -c \"import ctypes;ctypes.CDLL('/path/to/libtao_ops.so').print_tao_build_info()\"\n".format(
                PYTHON_BIN_NAME
            )
        )
        for k, v in sorted(os.environ.items()):
            if k.startswith("TAO_DOCKER_") or k.startswith("TAO_BUILD_"):
                f.write("{}={}\n".format(k, v))


def get_libstdcxx(gcc_ver):
    """
    Get full path of linked libstdc++.so and a normalized name in format
    libstdc++.so.[0-9] .
    """
    lib_path = VALID_GCC[gcc_ver][1]
    if not lib_path:
        return None, None
    pat = re.compile("^libstdc\+\+.so(\.[0-9]+)*$")
    for f in os.listdir(lib_path):
        full_path = os.path.join(lib_path, f)
        if os.path.isfile(full_path) and not os.path.islink(full_path) and pat.match(f):
            match = re.search("^libstdc\+\+.so\.[0-9]+", f)
            if match:
                normed_name = match.group(0)
                return full_path, normed_name
    logger.warn("No libstdc++ found for gcc " + gcc_ver + "on path: " + lib_path)
    return None, None


@time_stage()
def make_package(root, args):
    libstdcxx_path, libstdcxx_name = get_libstdcxx(args.compiler_gcc)
    if libstdcxx_path:
        logger.info(
            "Packaging libstdc++ (named %s): %s" % (libstdcxx_name, libstdcxx_path)
        )
    project_root = get_source_root_dir()
    with cwd(root):
        logger.info("packaging for version: " + args.version)
        build_info_file = "{}/built/tao/build.txt".format(root)
        generate_build_info(build_info_file)

        F_TAO_COMPILER_MAIN = (
            "./tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main"
        )
        F_TAO_OPS_SO = "./tao/bazel-bin/libtao_ops.so"
        F_PTXAS = "./tao/third_party/ptxas/10.2/ptxas"

        def add_to_tar(tar, file, dir_in_tar="", name_in_tar=""):
            name_in_tar = name_in_tar or os.path.basename(file)
            full_path_in_tar = os.path.join(dir_in_tar, name_in_tar)
            tar.add(file, arcname=full_path_in_tar)

        # pkg for dsw.
        dsw_tgz = "{}/built/tao/tao_dsw_{}.tgz".format(root, args.version)
        with tarfile.open(dsw_tgz, "w:gz") as tar:
            add_to_tar(tar, F_TAO_COMPILER_MAIN)
            add_to_tar(tar, F_TAO_OPS_SO)
            add_to_tar(tar, F_PTXAS)
            add_to_tar(tar, build_info_file)
            if libstdcxx_path:
                add_to_tar(tar, libstdcxx_path, name_in_tar=libstdcxx_name)
        logger.info("dsw package created   : " + dsw_tgz)

        logger.info("Stage [make_package] success.")

@time_stage()
def sanity_check(git_target="origin/master"):
    # Clang format for all h/c/cc file
    # This will only check the difference between current branch and the git target
    root = get_source_root_dir()
    clang_format_cmd = root + "../platform_alibaba/ci_build/lint/git-clang-format.sh " + git_target
    execute(clang_format_cmd)
    # TODO(): Add python lint later



def parse_args():
    # flag definition
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "venv_dir", help="Directory of virtualenv where target tensorflow installed."
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        default="auto",
        help="Version of built packages, defaults to %(default)s. auto: read from VERSION file.",
    )
    parser.add_argument(
        "--bridge-gcc",
        required=False,
        choices=VALID_GCC,
        help="GCC version to compile tao bridge, required for configure stages.",
    )
    parser.add_argument(
        "--compiler-gcc",
        required=False,
        choices=["7.3", "7.5", "default"],
        help="GCC version to compile tao compiler, required for configure stages.",
    )
    parser.add_argument(
        "--skip-tests",
        required=False,
        action="store_true",
        help="Skip all tests.",
    )
    parser.add_argument(
        "--skip-sanity",
        required=False,
        action="store_true",
        help="Skip all sanity checks.",
    )
    parser.add_argument(
        "--no-build-for-test",
        required=False,
        action="store_true",
        help="Skip build stages when running test stages. By default, "
        "build stages\nwill be triggered before test stages run.",
    )
    parser.add_argument(
        "-s",
        "--stage",
        required=False,
        choices=[
            "all",
            "configure",
            "configure_pytorch",
            "lint",
            "build",
            "build_tao_compiler",
            "build_tao_bridge",
            "build_dsw",
            "build_mlir_ral",
            "test",
            "test_bace",
            "test_tao_bridge_cpp",
            "test_tao_bridge_py",
            "test_tao_compiler",
            "package",
            "package_ral",
        ],
        default="all",
        metavar="stage",
        help="""Run all or a single build stage or sub-stage, it can be:
    - all: The default, it will configure, build, test and package.
    - configure: configure tao_compiler and tao_bridge for building.
    - lint: Run lint on c/c++/python codes.
    - build: parent stage of the following:
        - build_tao_compiler: build tao_compiler only.
        - build_tao_bridge: build tao_bridge only.
        - build_dsw: build dsw .whl only.
    - test: test tao_compiler (not ready for now) and tao_bridge.
        - test_tao_bridge_cpp: run cpp unit tests for tao_bridge.
        - test_tao_bridge_py: run python unit tests for tao_bridge.
        - test_tao_compiler: run unit tests for tao_compiler_main.
    - package: make release packages.""",
    )
    parser.add_argument(
        "--disable_link_tf_framework",
        required=False,
        action="store_true",
        help="Skip linking tf framework",
    )
    parser.add_argument(
        "--cpu_only",
        required=False,
        action="store_true",
        help="Build tao with cpu support only",
    )
    parser.add_argument(
        "--aarch64",
        required=False,
        action="store_true",
        help="Build tao with aarch64 support only",
    )
    parser.add_argument(
        "--dcu",
        required=False,
        action="store_true",
        help="Build tao with dcu support only",
    )
    parser.add_argument(
        "--build_in_aone",
        required=False,
        action="store_true",
        help="Build tao in aone env",
    )
    parser.add_argument(
        "--build_in_tf_addon",
        required=False,
        action="store_true",
        help="Build tao in tensorflow-addons tree",
    )
    parser.add_argument(
        "--ral_cxx11_abi",
        required=False,
        action="store_true",
        help="Build ral standalone lib with cxx11 abi or not",
    )
    parser.add_argument(
        "--enable_blaze_opt",
        required=False,
        action="store_true",
        help="Build tao with ad blaze optimization",
    )
    parser.add_argument(
        "--bazel_target",
        help="bazel build/test targets for tao compiler",
    )
    parser.add_argument(
        "--cmake",
        required=False,
        action="store_true",
        help="cmake build/test targets for tao bridge",
    )
    parser.add_argument(
        "--build_dbg_symbol", action="store_true", help="Add -g to build options"
    )
    parser.add_argument(
        "--platform_alibaba", action="store_true", help="build with is_platform_alibaba=True"
    )
    parser.add_argument(
        "--blade_gemm", default=True, action="store_true", help="build with is_blade_gemm=True"
    )
    # flag validation
    args = parser.parse_args()
    assert args.venv_dir, "virtualenv directory should not be empty."
    assert os.path.exists(args.venv_dir), "virtualenv directory does not exist."
    args.venv_dir = os.path.abspath(args.venv_dir)

    update_cpu_specific_setting(args)

    if args.stage in ["all", "configure"]:
        assert args.bridge_gcc, "--bridge-gcc is required."
        assert args.compiler_gcc, "--compiler-gcc is required."
    elif args.stage in ["configure_pytorch"]:
        assert args.bridge_gcc, "--bridge-gcc is required."
    else:
        assert (
            args.bridge_gcc is None
        ), "--bridge-gcc should be given only for configure."
        assert (
            args.compiler_gcc is None
        ), "--compiler-gcc should be given only for configure."

    if args.version == "auto":
        args.version = open(get_version_file()).read().split()[0]

    if args.platform_alibaba and args.blade_gemm:
        cuda_ver, _ = deduce_cuda_info()
        if '10\.' in cuda_ver:
            args.blade_gemm = False
    return args


def main():
    args = parse_args()
    root = get_source_root_dir()
    prepare_env(args)
    stage = args.stage

    if stage in ["lint"] and args.platform_alibaba:
        sanity_check()
        if stage == "lint": return

    # deal with aone env
    if args.build_in_aone:
        os.environ[
            "TF_BUILD_OPTS"
        ] = "--copt=-w --curses=no --color=no --noshow_loading_progress --noshow_progress"
        os.environ[
            "TF_TEST_OPTS"
        ] = "--copt=-w --curses=no --color=no --noshow_loading_progress --noshow_progress"
        logger.info(os.environ["TF_BUILD_OPTS"])
        logger.info(os.environ["TF_TEST_OPTS"])

    if stage in ["all", "configure", "configure_pytorch"]:
        if args.enable_mkldnn:
            with gcc_env(args.bridge_gcc):
                config_mkldnn(root, args)
        if stage == "configure_pytorch":
            configure_pytorch(root, args)
        else:
            configure(root, args)

    restore_gcc_conf(args)
    assert args.compiler_gcc in ["7.3", "7.5", "default"], "compiler_gcc {} not supported".format(
        args.compiler_gcc
    )

    is_test = stage in ["test", "test_tao_bridge_cpp", "test_tao_bridge_py"]

    if (
        stage in ["all", "build", "build_tao_compiler", "build_mlir_ral"]
        or is_test
        and not args.no_build_for_test
    ):
        if args.enable_mkldnn:
            with gcc_env(args.bridge_gcc):
                build_mkldnn(root)
        if stage == "build_mlir_ral":
            build_mlir_ral(root, args)
        else:
            build_tao_compiler(root, args)

    if (
        stage in ["all", "build", "build_tao_bridge"]
        or is_test
        and not args.no_build_for_test
    ):
        build_tao_bridge(root, args)

    if not args.skip_tests:
        if stage in ["all", "test", "test_tao_bridge_cpp"]:
            test_tao_bridge(root, args, cpp=True, python=False)

        if stage in ["all", "test", "test_tao_bridge_py"]:
            test_tao_bridge(root, args, cpp=False, python=True)

        if stage in ["all", "test", "test_tao_compiler"]:
            if args.enable_mkldnn:
                build_mkldnn(root)
            test_tao_compiler(root, args)

    if stage in ["all", "package"] and args.platform_alibaba:
        make_package(root, args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        stage_time.report()

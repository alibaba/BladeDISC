#!/usr/bin/env python3
# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# type: ignore

import argparse
import os
import random
import re
import subprocess

from six.moves import cPickle as pickle

from common_internal import (
    PY_VER,
    cwd,
    deduce_cuda_info,
    ensure_empty_dir,
    execute,
    get_cudnn_version,
    get_site_packages_dir,
    get_trt_version,
    logger,
    safe_run,
    which,
)

# Source code root dir.
ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
BUILD_CONFIG = os.path.join(ROOT, ".build_config")


def get_tf_info(python_dir):
    python_dir = os.path.abspath(python_dir)
    python_bin = os.path.join(python_dir, 'bin', 'python')
    output = subprocess.check_output(
        '{} -c "import tensorflow as tf; print(tf.__version__); print(\'\\n\'.join(tf.sysconfig.get_compile_flags())); print(\'\\n\'.join(tf.sysconfig.get_link_flags()))"'.format(
            python_bin
        ),
        shell=True,
    ).decode()
    lines = output.split("\n")
    major, minor, _ = lines[0].split(".")  # lines[0] is version like 1.15.0
    is_pai = "PAI" in lines[0]
    header_dir, lib_dir, lib_name, cxx11_abi = '', '', '', ''
    for line in lines[1:]:
        if line.startswith("-I"):
            header_dir = line[2:]
        elif line.startswith("-L"):
            lib_dir = line[2:]
        elif line.startswith("-l:"):  # in case of -l:libtensorflow_framework.so.1
            lib_name = line[3:]
        elif line.startswith("-l"):  # in case of -ltensorflow_framework
            lib_name = 'lib' + line[2:] + '.so'
        elif '_GLIBCXX_USE_CXX11_ABI' in line:
            cxx11_abi = line.split('=')[-1]
    return major, minor, is_pai, header_dir, lib_dir, lib_name, cxx11_abi


def save_build_config(args):
    arg_dict = dict(vars(args).items())
    arg_dict.pop('stage')
    with open(BUILD_CONFIG, 'wb') as f:
        pickle.dump(arg_dict, f)


def restore_build_config(args):
    if not os.path.exists(BUILD_CONFIG):
        return
    with open(BUILD_CONFIG, 'rb') as f:
        saved_dict = pickle.load(f)
    for k in args.__dict__:
        if k in saved_dict.keys():
            args.__dict__[k] = saved_dict[k]


def get_test_tag_filters(args, tf_major=None):
    if args.device == "cpu":
        config = "--test_tag_filters=-gpu"
    elif args.device == "gpu":
        config = "--test_tag_filters=-cpu"

    if tf_major is None and args.tf:
        tf_major = args.tf.split('.')[0]
    if tf_major == "2":
        config += ",-tf1"  # skip tf1-only tests when it's tf2.
    elif tf_major == "1":
        config += ",-tf2"  # skip tf2-only tests when it's tf1.
    return config


def check_init_file_miss(path, ignore_root=False):
    path = os.path.abspath(path)
    for dir, _, _ in os.walk(path):
        if ignore_root and path == dir:
            continue
        name = os.path.basename(dir)
        # skip checking for special dirs
        if name.startswith(".") or name.startswith("_"):
            continue
        # skip build directories
        if (
            "/build" in dir
            or "/dist" in dir
            or name == 'lib'
            or name.endswith(".egg-info")
        ):
            continue
        if not os.path.exists(os.path.join(dir, '__init__.py')):
            raise Exception("missing __init__.py under " + dir)


# No need to do cc check, since pre-commit do cc check with clang-format
def check(args):
    with cwd(ROOT):
        # every folder under python should contain a __init__.py file
        # check tests dir
        check_init_file_miss("tests", ignore_root=True)
        execute("{}/bin/black --check --diff tests".format(args.python_dir))
        execute("{}/bin/flake8 tests".format(args.python_dir))
        execute("{}/bin/mypy tests".format(args.python_dir))


def configure_with_bazel(args):
    save_build_config(args)
    with open(os.path.join(ROOT, ".bazelrc_gen"), "w") as f:

        def _opt(opt, value, cmd="build"):
            f.write(f"{cmd} --{opt}={value}\n")

        def _action_env(key, value, cmd="build"):
            f.write(f"{cmd} --action_env {key}={value}\n")

        def _write(line, cmd="build"):
            f.write(f"{cmd} {line}\n")

        # Common
        _opt("cxxopt", "-std=c++14")
        _opt("host_cxxopt", "-std=c++14")
        _opt("compilation_mode", "opt")
        _opt("cxxopt", "-DBUILD_WITH_BAZEL")
        _action_env("BUILD_WITH_BAZEL", "1")
        _action_env("PYTHON_BIN_PATH", os.path.join(args.python_dir, 'bin', 'python'))

        site_packages_dir = os.path.join(get_site_packages_dir(args.python_dir))

        (
            tf_major,
            tf_minor,
            is_pai,
            tf_header_dir,
            tf_lib_dir,
            tf_lib_name,
            tf_cxx11_abi,
        ) = get_tf_info(args.python_dir)
        _action_env("BLADE_WITH_TF", "1")
        _opt("cxxopt", f"-D_GLIBCXX_USE_CXX11_ABI={tf_cxx11_abi}")
        _opt("host_cxxopt", f"-D_GLIBCXX_USE_CXX11_ABI={tf_cxx11_abi}")
        _action_env("TF_IS_PAI", int(is_pai))
        _action_env("TF_MAJOR_VERSION", tf_major)
        _action_env("TF_MINOR_VERSION", tf_minor)
        _action_env("TF_HEADER_DIR", tf_header_dir)
        _action_env("TF_SHARED_LIBRARY_DIR", tf_lib_dir)
        _action_env("TF_SHARED_LIBRARY_NAME", tf_lib_name)

        # TF-Blade
        _action_env("BLADE_WITH_TF_BLADE", "1")
        _action_env("BLADE_WITH_INTERNAL", "1" if args.internal else "0")

        # CUDA
        if args.device == "gpu":
            cuda_ver, cuda_home = deduce_cuda_info()
            cudnn_ver = get_cudnn_version(cuda_home)
            # Following tf community's cuda related action envs
            _action_env("TF_NEED_CUDA", "1")
            _action_env("GCC_HOST_COMPILER_PATH", which("gcc"))
            _action_env("TF_CUDA_CLANG", "0")
            _action_env("TF_CUDA_VERSION", cuda_ver)
            _action_env("TF_cuda_homeS", cuda_home)
            _action_env("TF_CUDNN_VERSION", cudnn_ver)
            if '11\.' in cuda_ver:
                _action_env("TF_CUDA_COMPUTE_CAPABILITIES", "7.0,7.5,8.0")
            elif '10\.' in cuda_ver:
                _action_env("TF_CUDA_COMPUTE_CAPABILITIES", "7.0,7.5")
            _action_env("NVCC", which("nvcc"))
            _opt("define", "using_cuda=true")
            _write("--@local_config_cuda//:enable_cuda")
            _write("--crosstool_top=@local_config_cuda//crosstool:toolchain")

            if not args.skip_trt:
                _action_env("BLADE_WITH_TENSORRT", "1")
                trt_root = os.environ.get("TENSORRT_INSTALL_PATH", "/usr/local/TensorRT")
                _action_env("TENSORRT_VERSION", get_trt_version(trt_root))
                _action_env("TENSORRT_INSTALL_PATH", trt_root)
            else:
                _action_env("BLADE_WITH_TENSORRT", "0")

            _action_env("BLADE_WITH_HIE", "1" if args.internal and not args.skip_hie else "0")

            _write("--//:device=gpu")
            _action_env("BLADE_WITH_MKL", "0")
        else:
            _action_env("TF_NEED_CUDA", "0")
            _action_env("BLADE_WITH_TENSORRT", "0")
            _action_env("BLADE_WITH_HIE", "0")
            _opt("define", "using_cuda=false")
            _write("--//:device=cpu")
            if args.device == 'cpu':
                # TODO(lanbo.llb): unify mkl configure with tao_bridge
                _action_env("BLADE_WITH_MKL", "1")
                mkl_root = os.environ.get("MKL_INSTALL_PATH", "/opt/intel/compilers_and_libraries_2020.1.217/linux")
                assert os.path.exists(mkl_root), f"MKL root path missing: {mkl_root}"
                _action_env("MKL_INSTALL_PATH", mkl_root)

        _write(f"--//:framework=tf")
        _write(
            get_test_tag_filters(args, tf_major=tf_major), cmd="test",
        )
        # Working around bazel #10327
        _action_env("BAZEL_LINKOPTS", os.environ.get("BAZEL_LINKOPTS", ""))
        _action_env("BAZEL_LINKLIBS", os.environ.get("BAZEL_LINKLIBS", "-lstdc++"))
    logger.info("Writing to .bazelrc_gen done.")


def build_with_bazel(args):
    with cwd(ROOT):
        execute("bazel build //src:_tf_blade.so")

def package_whl_with_bazel(args):
    with cwd(ROOT):
        execute("bazel run //:build_pip_package")
        dist_dir = os.path.join(ROOT, 'dist')
        build_dir = os.path.join(
            ROOT,
            'bazel-bin',
            'build_pip_package.runfiles',
            'org_tf_blade',
            'dist',
        )
        ensure_empty_dir(dist_dir)
        execute(f"mv {build_dir}/*.whl {dist_dir}")

def test_with_bazel(args):
    with cwd(ROOT):
        execute("bazel test //tests/...")
    logger.info("Stage [test] success.")


def parse_args():
    # flag definition
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "python_dir",
        help="""
            Directory of virtualenv where target tensorflow installed.
            If not specificed, the default python will be used(from `which python3`).
        """
    )
    parser.add_argument(
        "-s",
        "--stage",
        required=False,
        choices=[
            "check",
            "configure",
            "build",
            "test",
            "package",
        ],
        default="all",
        metavar="stage",
        help="""Run all or a single build stage or sub-stage, it can be:
    - all: The default, it will configure, build, test and package.
    - check: Run format checkers and static linters.
    - configure: parent stage of the following:
    - build: build tf blade with regard to the configured part
    - test: test tf blade framework.
    - package: make tf blade python packages."""
    )
    parser.add_argument(
        "--device",
        required=False,
        default="gpu",
        choices=["cpu", "gpu"],
        help='Build target device',
    )
    parser.add_argument(
        "--tf", required=False, choices=["1.15", "2.4"], help="Tensorflow version.",
    )
    parser.add_argument(
        '--skip_trt',
        action="store_true",
        required=False,
        default=False,
        help="If True, tensorrt will be skipped for gpu build",
    )
    parser.add_argument(
        '--skip_hie',
        action="store_true",
        required=False,
        default=True,
        help="If True, hie will be skipped for internal build",
    )
    parser.add_argument(
        '--internal',
        action="store_true",
        required=False,
        default=False,
        help="If True, internal objects will be built",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Show more information in each stage",
    )

    # flag validation
    args = parser.parse_args()
    args.python_dir = os.path.abspath(args.python_dir)
    return args


def setup_env():
    if "BLADE_CI_ENV" not in os.environ:
        return
    nv_smi_cmd = "/usr/local/nvidia/bin/nvidia-smi"
    if not os.path.exists(nv_smi_cmd):
        logger.warning(
            "Skip choosing random CUDA divice since {} is not found.".format(nv_smi_cmd)
        )
        return
    out = safe_run(nv_smi_cmd + " -L | wc -l", shell=True)
    num_gpu = int(out.strip())
    idx = random.randint(0, num_gpu - 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    logger.info("Choose GPU {} among {} GPUs in total.".format(idx, num_gpu))


def main():
    args = parse_args()
    setup_env()

    stage = args.stage
    if stage in ["all", "check"]:
        check(args)

    if stage in ["all", "configure"]:
        configure_with_bazel(args)

    restore_build_config(args)
    if stage in ["all", "build"]:
        build_with_bazel(args)

    if stage in ["all", "test"]:
        test_with_bazel(args)

    if stage in ["all", "package"]:
        package_whl_with_bazel(args)

if __name__ == "__main__":
    main()

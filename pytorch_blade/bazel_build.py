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

import argparse
import os
import errno
import subprocess
import sys
import torch
import venv

from common_setup import (
    running_on_ci,
    remote_cache_token,
    which,
    num_make_jobs,
    is_aarch64,
    build_tao_compiler_add_flags_platform_alibaba_cached,
    test_tao_compiler_add_flags_platform_alibaba_cached,
)
from torch_blade_build import TorchBladeBuild, get_fullpath_or_create

cwd = os.path.dirname(os.path.abspath(__file__))


def _make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def _symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


class BazelBuild(TorchBladeBuild):

    def pybind11_cflags(self):
        # see https://github.com/pytorch/pytorch/blob/fe87ae692f813934d1a74d000fd1e3b546c27ae2/torch/utils/cpp_extension.py#L515
        common_cflags = []
        if self.torch_major_version == 1 and self.torch_minor_version <= 6:
            return common_cflags
        for pname in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            pval = getattr(torch._C, f"_PYBIND11_{pname}")
            if pval is not None:
                common_cflags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')
        return common_cflags

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = [
            "@org_disc_compiler//mlir/xla/ral:libral_base_context.so",
            "//pytorch_blade:libtorch_blade.so",
            "//pytorch_blade:_torch_blade.so",
            "//tests/mhlo/torch-mlir-opt:torch-mlir-opt",
            "//tests/torchscript:shape_analysis_tool",
            "//tests/torch-disc-pdll:torch-disc-pdll",
        ]

        torch_major_version, torch_minor_version = self.torch_version.split(".")[:2]
        self.torch_major_version = int(torch_major_version)
        self.torch_minor_version = int(torch_minor_version)

        # ----------------------------------------------------------------------- #
        # ------------  Basic Settings for DISC: disc_base_opts   --------------- #
        # ----------------------------------------------------------------------- #
        self.disc_base_opts = [
            "--action_env PYTHON_BIN_PATH={}".format(sys.executable),
            # Workaroud issue: https://github.com/bazelbuild/bazel/issues/10327
            "--action_env BAZEL_LINKLIBS=-lstdc++",
            "--action_env CC={}".format(which("gcc")),
            "--action_env CXX={}".format(which("g++")),
            "--action_env DISC_FOREIGN_MAKE_JOBS={}".format(num_make_jobs())
        ]

        remote_cache = remote_cache_token()
        if remote_cache:
            self.disc_base_opts += ["--remote_cache={}".format(remote_cache)]

        # ----------------------------------------------------------------------- #
        # ------   Extra Settings for Torch Frontend: torch_extra_opts    ------- #
        # ----------------------------------------------------------------------- #
        self.torch_extra_opts = [
            '--copt=-DPYTORCH_VERSION_STRING=\\"{}\\"'.format(self.torch_version),
            "--copt=-DPYTORCH_MAJOR_VERSION={}".format(torch_major_version),
            "--copt=-DPYTORCH_MINOR_VERSION={}".format(torch_minor_version),
            "--copt=-DTORCH_BLADE_CUDA_VERSION={}".format(self.cuda_version),
            "--action_env TORCH_BLADE_TORCH_INSTALL_PATH={}".format(self.torch_dir),
        ] + ['--copt={}'.format(cflag) for cflag in self.pybind11_cflags()]

        if (self.torch_major_version, self.torch_minor_version) == (1, 12):
            # LTC features only tested on torch==1.12.0+cu113 for now
            self.torch_extra_opts.append("--config=torch_ltc_disc_backend")
        if self.is_debug:
            self.torch_extra_opts.append("--config=torch_debug")
        else:
            self.torch_extra_opts.append("--compilation_mode=opt")

        if self.cuda_available and self.build_tensorrt:
            self.torch_extra_opts.append(
                "--config=torch_static_tensorrt"
                if self.static_tensorrt
                else "--config=torch_tensorrt"
            )
            self.torch_extra_opts += [
                "--action_env TENSORRT_INSTALL_PATH={}".format(self.tensorrt_dir),
                "--action_env NVCC={}".format(which("nvcc")),
            ]

        # ----------------------------------------------------------------------- #
        # ---------------    Configurations Settings: configs    ---------------- #
        # ----------------------------------------------------------------------- #
        self.configs = [
            "--config=torch_cxx11abi_{}".format(int(self.GLIBCXX_USE_CXX11_ABI))
        ]
        if self.cuda_available:
            self.configs.append("--config=torch_cuda")

        elif self.dcu_rocm_available:
            self.configs.append("--config=torch_dcu_rocm")

        else:
            if is_aarch64():
                self.configs += ["--config=torch_aarch64"]
            else:
                self.configs += ["--config=torch_x86"]

        if self.cuda_available and float(self.cuda_version) >= 11.0 and self.blade_gemm:
            self.configs += ["--config=blade_gemm"]

        if self.build_hie:
            print("[WARNIGN] HIE will be disabled temporarily.")
            # self.configs += ["--config=hie"]

        if self.skip_compute_intensive_fusion:
            self.configs += ["--config=skip_compute_intensive_fusion"]

        if running_on_ci():
            self.configs += ["--config=ci_build"]

        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        self.configs += [
            build_tao_compiler_add_flags_platform_alibaba_cached(root_dir, ""),
            test_tao_compiler_add_flags_platform_alibaba_cached(root_dir, "")
        ]

        # ----------------------------------------------------------------------- #
        # --------------------   Settings for Quantization   -------------------- #
        # ----------------------------------------------------------------------- #
        # This is for quantization process starts from blade_compression.
        # blade_compression uses fx graph to do quantization, which is
        # introduced in torch 1.8.0. So the lower bound of torch version is
        # set to 1.8.0
        # note: If we use backend (such as trt) to do calibration and get the
        # quantization engine, this limitation should not be set.
        is_enable_quantization = (
            self.torch_major_version == 1
            and self.torch_minor_version >= 8
            or self.torch_major_version > 1
        )
        if is_enable_quantization:
            self.torch_extra_opts.append("--config=torch_enable_quantization")

        # ----------------------------------------------------------------------- #
        # --------------------   Settings for Neural Engine   ------------------- #
        # ----------------------------------------------------------------------- #
        if self.build_neural_engine:
            print("=================enable neural engine=============")
            self.torch_extra_opts.append("--config=torch_enable_neural_engine")

        self.shell_setting = "set -e; set -o pipefail; "
        # Workaround: this venv ensure that $(/usr/bin/env python) is evaluated to python3
        venv.create(".bazel_pyenv", clear=True)
        self.build_cmd = "source .bazel_pyenv/bin/activate; bazel build --verbose_failures"
        self.test_cmd = "source .bazel_pyenv/bin/activate; bazel test"
        if running_on_ci():
            self.test_cmd += " --test_output=errors"

    def run(self, extdir=None, srcdir=None, build_temp=None):
        srcdir = get_fullpath_or_create(
            srcdir or os.path.dirname(os.path.abspath(__file__))
        )
        extdir = get_fullpath_or_create(extdir or "build/temp")
        bazel_bin_dir = os.path.join(srcdir, "bazel-bin/")

        env = os.environ.copy()
        ld_library_path = ":".join([self.torch_lib_dir, env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path
        env["GCC_HOST_COMPILER_PATH"] = env.get("GCC_HOST_COMPILER_PATH", which("gcc"))

        # 1. build disc_compiler_main, only needs basic opts and configs
        # FIXME: debug mode compilation would failed for now
        if not self.skip_disc_cmd_build:
            bazel_disc_build_cmd = " ".join(
                [self.shell_setting, self.build_cmd]
                + self.disc_base_opts
                + self.configs
                + [
                    "--compilation_mode=opt",
                    "--define is_torch_disc=false",  # still use mhlo within TF to build compiler main
                    "@org_disc_compiler//mlir/disc:disc_compiler_main",
                ]
            )
            subprocess.check_call(
                bazel_disc_build_cmd, shell=True, env=env, executable="/bin/bash"
            )

        # 2. build other targets, support both debug mode & opt mode compilation
        bazel_cmd = " ".join(
            [self.shell_setting, self.build_cmd]
            + self.disc_base_opts
            + self.torch_extra_opts
            + self.configs
        )
        with open("debug_bazel.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("export LD_LIBRARY_PATH={}\n".format(ld_library_path))
            f.write(
                "export GCC_HOST_COMPILER_PATH={}\n".format(
                    env.get("GCC_HOST_COMPILER_PATH", "")
                )
            )
            f.write(bazel_cmd + " $@")
        _make_executable("debug_bazel.sh")

        bazel_cmd = " ".join([bazel_cmd] + self.targets)
        subprocess.check_call(bazel_cmd, shell=True, env=env, executable="/bin/bash")

        # If you want to package more files, please extends the distribution.cfg.
        # We symlink those files into extension's directory, since that
        # python distribute utils will copy into the distribution package.
        #
        # Note that only the following file pathes would be accepted:
        # 1. file pathes relevent to your bazel bin directory
        # 2. absolute file pathes
        for fpath in open("distribution.cfg"):
            fpath = os.path.realpath(os.path.join(bazel_bin_dir, fpath.strip()))
            fname = os.path.basename(fpath)
            if os.path.exists(fpath):
                _symlink_force(fpath, os.path.join(extdir, fname))
            else:
                print(f"{fpath} configured to distribution doesn't exists")

    def test(self):
        env = os.environ.copy()
        ld_library_path = ":".join([self.torch_lib_dir, env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path
        env["GCC_HOST_COMPILER_PATH"] = env.get("GCC_HOST_COMPILER_PATH", which("gcc"))

        self.test_suites = [
            "//tests/mhlo/...",
            "//pytorch_blade:torch_blade_test_suite",
            "//tests/torch-disc-pdll/tests/...",
        ]

        if (self.torch_major_version, self.torch_minor_version) > (1, 6):
            # torchscript graph ir parser changed after torch 1.6.
            # We will not test torchscript graph ir before torch 1.6
            self.test_suites.append("//tests/torchscript/...")

        test_cmd = " ".join(
            [self.shell_setting, self.test_cmd]
            + self.disc_base_opts
            + self.torch_extra_opts
            + self.configs
            + self.test_suites
        )
        subprocess.check_call(test_cmd, shell=True, env=env, executable="/bin/bash")

        self.disc_test_suites = [
            "//tests/disc_mlir/...",
        ]
        disc_test_extra_opts = ['--config=disc_test']
        if self.is_debug:
            # The config opt disc_test_debug is not rational but without it we will run into the following issue:
            # In file included from /usr/include/stdint.h:25,
            #                  from /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/include/stdint.h:9,
            #                  from external/boringssl/src/include/openssl/base.h:60,
            #                  from external/boringssl/err_data.c:17:
            # /usr/include/features.h:330:4: error: #warning _FORTIFY_SOURCE requires compiling with optimization (-O) [-Werror=cpp]
            disc_test_extra_opts.append("--config=disc_test_debug")

        disc_test_cmd = " ".join(
            [self.shell_setting, self.test_cmd]
            + self.disc_base_opts
            + disc_test_extra_opts
            + self.configs
            + self.disc_test_suites
        )
        subprocess.check_call(disc_test_cmd, shell=True, env=env, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bazel build TorchBlade")
    parser.add_argument(
        "--torch_version", type=str, required=True, help="The version of torch"
    )
    parser.add_argument(
        "--torch_dir", type=str, required=True, help="The directory where torch located"
    )
    parser.add_argument(
        "--cuda_version", type=str, default=None, help="The version of cuda toolkit"
    )
    parser.add_argument("--cxx11", action="store_true", help="Use c++ cxx11 abi")

    args = parser.parse_args()

    build = BazelBuild(
        args.torch_dir, args.torch_version, args.cuda_version, cxx11_abi=args.cxx11
    )
    build.write_version_file(os.path.join(cwd, "version.txt"))
    srcdir = os.path.dirname(os.path.abspath(__file__))
    build.run(extdir=os.path.join(srcdir, "torch_blade"))

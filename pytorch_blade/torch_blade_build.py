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

import os
import subprocess
import shutil
import re
import release_version

cwd = os.path.dirname(os.path.abspath(__file__))


def check_env_flag(name, default=None):
    env_val = os.getenv(name, default)
    return (
        None if env_val is None else env_val.upper() in ["ON", "1", "YES", "TRUE", "Y"]
    )


def get_fullpath_or_create(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.abspath(dir_path)


class TorchBladeBuild:
    def __init__(
        self,
        torch_dir,
        torch_version,
        cuda_version=None,
        torch_git_version=None,
        cxx11_abi=False,
    ):
        self.__serialization_version__ = "0.0.3"
        self.torch_dir = torch_dir
        self.torch_lib_dir = os.path.join(self.torch_dir, 'lib')
        self.cuda_version = cuda_version
        self.torch_version = torch_version
        self.git_version = self.get_git_version()
        self.torch_git_version = torch_git_version
        self.GLIBCXX_USE_CXX11_ABI = cxx11_abi
        # NB: Bump up because of MLIR Engine serialization changes
        self.is_debug = check_env_flag("DEBUG", "OFF")
        self.cuda_available = check_env_flag(
            "TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT", "ON"
        )
        self.dcu_rocm_available = check_env_flag( 
            "TORCH_BLADE_BUILD_WITH_DCU_ROCM_SUPPORT", "OFF"    
        )
        self.build_tensorrt = check_env_flag(
            "TORCH_BLADE_BUILD_TENSORRT", "OFF"
        )
        self.static_tensorrt = check_env_flag(
            "TORCH_BLADE_BUILD_TENSORRT_STATIC", "OFF"
        )
        self.blade_gemm = check_env_flag(
            "TORCH_BLADE_BUILD_BLADE_GEMM", "OFF"
        )
        self.skip_disc_cmd_build = check_env_flag(
            "TORCH_BLADE_SKIP_DISC_CMD_BUILD", "OFF"
        )
        self.build_hie = check_env_flag(
            "TORCH_BLADE_BUILD_HIE", "OFF"
        )
        self.skip_compute_intensive_fusion = check_env_flag(
            "TORCH_BLADE_BUILD_SKIP_COMPUTE_INTENSIVE_FUSION", "OFF"
        )
        self.build_neural_engine = check_env_flag(
            "TORCH_BLADE_ENABLE_NEURAL_ENGINE", "OFF"
        )


        self.tensorrt_dir = os.getenv("TENSORRT_INSTALL_PATH", "/usr/local/TensorRT/")
        self.version = self.get_version()

    def get_git_version(self):
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode("ascii")
                .strip()
            )
        except Exception as e:
            print("\t", e)
            sha = "unknown"
        return sha

    def get_version(self):
        version = release_version.__version__
        if version == "0.0.0":
            # this is develop version
            version += ".dev0"

        if not self.cuda_version == "10.0":
            torch_ver = self.torch_version.split("+")[0]
            if self.cuda_version:
                cuda_ver = self.cuda_version.replace(".", "")
                torch_ver += f".cu{cuda_ver}"

            version += f"+{torch_ver}"
        return version

    def write_version_file(self, version_path):
        with open(version_path, "w") as f:
            f.write("__version__ = {}\n".format(repr(self.version)))
            f.write(
                "__serialization_version__ = {}\n".format(
                    repr(self.__serialization_version__)
                )
            )
            f.write("debug = {}\n".format(repr(self.is_debug)))
            f.write("cuda = {}\n".format(repr(self.cuda_version)))
            f.write("cuda_available = {}\n".format(repr(self.cuda_available)))
            f.write("build_tensorrt = {}\n".format(repr(self.build_tensorrt)))
            f.write("static_tensorrt = {}\n".format(repr(self.static_tensorrt)))
            f.write("git_version = {}\n".format(repr(self.git_version)))
            f.write("torch_version = {}\n".format(repr(self.torch_version)))
            f.write("torch_git_version = {}\n".format(repr(self.torch_git_version)))
            f.write(
                "GLIBCXX_USE_CXX11_ABI = {}\n".format(repr(self.GLIBCXX_USE_CXX11_ABI))
            )

        with open(version_path, "r") as f:
            print("".join(f.readlines()))

    def patchelf_fix_sonames(self, extdir):
        if self.cuda_version is None:
            return
        if check_env_flag("TORCH_BLADE_DISABLE_PATCHELF_CUDA_SONAMES"):
            return

        torch_libs_dir = os.path.join(self.torch_dir, "lib")
        cuda_ver = self.cuda_version
        deps_lib_patch_regex = {
            f"libnvrtc.so.{cuda_ver}": f"libnvrtc(-[0-9a-fA-F]+)?.so.{cuda_ver}",
            f"libcudart.so.{cuda_ver}": f"libcudart(-[0-9a-fA-F]+)?.so.{cuda_ver}",
            "libnvToolsExt.so.1": "libnvToolsExt(-[0-9a-fA-F]+)?.so.1",
        }
        libs = [os.path.basename(lib) for lib in os.listdir(torch_libs_dir)]
        # search and save patchelf sonames from torch libraries if exists
        deps_patch_lib = dict()
        for deps_lib, patch_regex in deps_lib_patch_regex.items():
            regex = re.compile(patch_regex)
            for lib in libs:
                if regex.match(lib):
                    deps_patch_lib[deps_lib] = lib
                    shutil.copy(
                        os.path.join(torch_libs_dir, lib), os.path.join(extdir, lib)
                    )

        torch_blade_libs = [
            os.path.join(extdir, lib)
            for lib in os.listdir(extdir)
            if "libtorch_blade" in lib
        ]
        # use patchelf to replace needed sonames
        for origin_soname, patch_soname in deps_patch_lib.items():
            for lib in torch_blade_libs:
                subprocess.check_output(
                    ["patchelf", "--replace-needed", origin_soname, patch_soname, lib]
                )

    def run(self, extdir=None, srcdir=None, build_temp=None):
        pass

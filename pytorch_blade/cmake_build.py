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

import argparse
import os
import sys
import subprocess
import platform
import shutil
import re
from distutils.spawn import find_executable
from torch_blade_build import TorchBladeBuild, get_fullpath_or_create, check_env_flag

cwd = os.path.dirname(os.path.abspath(__file__))


class CMakeBuild(TorchBladeBuild):
    def run(self, extdir=None, srcdir=None, build_temp=None):
        extdir = get_fullpath_or_create(extdir or "build/temp")
        srcdir = get_fullpath_or_create(
            srcdir or os.path.dirname(os.path.abspath(__file__))
        )
        build_temp = get_fullpath_or_create(build_temp or extdir)
        build_temp = build_temp or extdir

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the project")

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        self.build_extension(extdir, srcdir, build_temp)
        try:
            self.patchelf_fix_sonames(extdir)
        except Exception:
            pass

    def build_extension(self, extdir, srcdir, build_temp):
        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
        torch_major_version, torch_minor_version = self.torch_version.split(".")[:2]
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPYTHON_VERSION=" + py_version,
            # PyTorch cmake args
            "-DPYTORCH_VERSION_STRING=" + self.torch_version,
            "-DPYTORCH_MAJOR_VERSION=" + torch_major_version,
            "-DPYTORCH_MINOR_VERSION=" + torch_minor_version,
            "-DPYTORCH_DIR={}".format(self.torch_dir),
            "-DTORCH_BLADE_USE_CXX11_ABI={}".format(self.GLIBCXX_USE_CXX11_ABI),
            "-DTORCH_BLADE_CUDA_VERSION={}".format(self.cuda_version),
            "-DTORCH_BLADE_BUILD_WITH_CUDA_SUPPORT={}".format(self.cuda_available),
        ]
        ccache = os.environ.get("CCACHE", None)
        executable = (
            ccache is not None and os.path.exists(ccache) and os.access(ccache, os.X_OK)
        )
        if executable:
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=%s" % ccache,
                "-DCMAKE_CXX_COMPILER_LAUNCHER=%s" % ccache,
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=%s" % ccache,
            ]

        # Use ninja if it's available
        if find_executable("ninja"):
            cmake_args.append("-GNinja")

        cfg = "Debug" if self.is_debug else "Release"
        build_args = ["--config", cfg, "--target", "package"]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j%d" % os.cpu_count()]

        def get_cmake_env_arg(env_var, default_val):
            env_val = check_env_flag(env_var, default_val)
            return [] if env_val is None else [f"-D{env_var}={env_val}"]

        cmake_args += get_cmake_env_arg("TORCH_BLADE_BUILD_MLIR_SUPPORT", None)
        cmake_args += get_cmake_env_arg("TORCH_BLADE_BUILD_PYTHON_SUPPORT", None)
        cmake_args += get_cmake_env_arg("TORCH_BLADE_PLATFORM_ALIBABA", None)

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.version
        )
        subprocess.check_call(["cmake", srcdir] + cmake_args, cwd=build_temp, env=env)
        shutil.copyfile(
            os.path.join(build_temp, "compile_commands.json"),
            os.path.join(srcdir, "compile_commands.json"),
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, env=env
        )
        with open("cpp_test.sh", "w") as out_f:
            out_f.write('cd {} && ctest "$@"\n'.format(build_temp))

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

    def test(self):
        if os.path.exists("cpp_test.sh"):
            self._run(["sh", "cpp_test.sh"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMake build TorchBlade")
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

    build = CMakeBuild(
        args.torch_dir, args.torch_version, args.cuda_version, cxx11_abi=args.cxx11
    )
    build.write_version_file(os.path.join(cwd, "version.txt"))
    build.run()

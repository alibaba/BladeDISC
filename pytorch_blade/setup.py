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

# Welcome to the PyTorch blade setup.py
#
# Environment variables that is optional:
#
#    DEBUG
#      build with -O0 and -g (debug symbols)
#

import os
import sys
import subprocess

from setuptools import find_packages
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
import bazel_build
import cmake_build

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
default_project_name = "torch_blade"
wheel_name = os.environ.get("TORCH_BLADE_WHL_NAME", default_project_name)
project_name = wheel_name

try:
    import torch
except ImportError as e:
    print("Unable to import torch. Error:")
    print("\t", e)
    print("You need to install pytorch first.")
    sys.exit(1)


class TorchBladeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

if '--cmake' in sys.argv:
  BuildClass = cmake_build.CMakeBuild
else:
  BuildClass = bazel_build.BazelBuild

build = BuildClass(
    os.path.dirname(torch.__file__),
    torch.version.__version__,
    cuda_version=torch.version.cuda,
    cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    torch_git_version=torch.version.git_version,
)

# In order to be packed into wheel, version.py must be created before
# setuptools.setup was called
build.write_version_file(os.path.join(cwd, project_name, "version.py"))


class TorchBladeBuild(build_ext):
    def run(self):
        torch_dir = os.path.dirname(torch.__file__)
        # version.txt Would be package into C++ SDK by CPACK
        build.write_version_file(os.path.join(cwd, "version.txt"))
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            build.run(extdir=extdir, srcdir=ext.sourcedir, build_temp="build/temp")


class CustomCommand(Command):
    user_options = []

    def initialize_options(self):
        # must override abstract method
        pass

    def finalize_options(self):
        # must override abstract method
        pass

class TestCommand(CustomCommand):

    def py_run(self):
        self._run(["python3", "-m", "unittest", "discover", "-v", "tests"])

    def cpp_run(self):
        build.test()

    def run(self):
        self.cpp_run()
        self.py_run()

    def _run(self, command):
        try:
            subprocess.check_call(command, shell=True, executable="/bin/bash")
        except subprocess.CalledProcessError as error:
            print("Command failed with exit code", error.returncode)
            sys.exit(error.returncode)


class CppTestCommand(TestCommand):
    def run(self):
        self.cpp_run()


class PyTestCommand(TestCommand):
    def run(self):
        self.py_run()

class BuildDepsCommand(CustomCommand):

    def run(self):
        cmd = "python3 ../scripts/python/common_setup.py"
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            cmd += " --cxx11_abi"

        if not build.cuda_available:
            cmd += " --build_mkldnn"

        subprocess.check_call(cmd, shell=True, executable="/bin/bash")


install_requires = ["networkx", "onnx>=1.6.0", f"torch=={torch.__version__}"]

wheel_suffix = "" if build.cuda_available else "-cpu"

setup(
    name=wheel_name + wheel_suffix,
    version=build.version,
    author="Alibaba PAI Team",
    description="The pytorch blade project",
    install_requires=install_requires,
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=[TorchBladeExtension("{}._torch_blade".format(wheel_name))],
    cmdclass=dict(
        build_ext=TorchBladeBuild,
        test=TestCommand,
        py_test=PyTestCommand,
        cpp_test=CppTestCommand,
        build_deps=BuildDepsCommand,
    ),
    zip_safe=False,
)

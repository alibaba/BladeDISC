# Welcome to the PyTorch addons setup.py
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
import cmake_build

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
default_project_name = "torch_addons"
wheel_name = os.environ.get("TORCH_ADDONS_WHL_NAME", default_project_name)
project_name = wheel_name

try:
    import torch
except ImportError as e:
    print("Unable to import torch. Error:")
    print("\t", e)
    print("You need to install pytorch first.")
    sys.exit(1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


build = cmake_build.CMakeBuild(
    os.path.dirname(torch.__file__),
    torch.version.__version__,
    cuda_version=torch.version.cuda,
    cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    torch_git_version=torch.version.git_version,
)

# In order to be packed into wheel, version.py must be created before
# setuptools.setup was called
build.write_version_file(os.path.join(cwd, project_name, "version.py"))


class CMakeBuild(build_ext):
    def run(self):
        torch_dir = os.path.dirname(torch.__file__)
        # version.txt Would be package into C++ SDK by CPACK
        build.write_version_file(os.path.join(cwd, "version.txt"))
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(
                                     self.get_ext_fullpath(ext.name)))
            build.run(extdir=extdir,
                      srcdir=ext.sourcedir,
                      build_temp="build/temp")


class TestCommand(Command):

    user_options = []

    def initialize_options(self):
        # must override abstract method
        pass

    def finalize_options(self):
        # must override abstract method
        pass

    def py_run(self):
        self._run(["py.test", "tests"])

    def cpp_run(self):
        if os.path.exists("cpp_test.sh"):
            self._run(["sh", "cpp_test.sh"])

    def run(self):
        self.cpp_run()
        self.py_run()

    def _run(self, command):
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as error:
            print("Command failed with exit code", error.returncode)
            sys.exit(error.returncode)


class CppTestCommand(TestCommand):
    def run(self):
        self.cpp_run()


class PyTestCommand(TestCommand):
    def run(self):
        self.py_run()


install_requires = ["networkx", "onnx>=1.6.0", f"torch=={torch.__version__}"]

wheel_suffix = "" if build.cuda_available else "-cpu"

setup(
    name=wheel_name + wheel_suffix,
    version=build.version,
    author="Alibaba PAI Team",
    description="The pytorch addons project",
    install_requires=install_requires,
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=[CMakeExtension("{}._torch_addons".format(wheel_name))],
    cmdclass=dict(
        build_ext=CMakeBuild,
        test=TestCommand,
        py_test=PyTestCommand,
        cpp_test=CppTestCommand,
    ),
    zip_safe=False,
)

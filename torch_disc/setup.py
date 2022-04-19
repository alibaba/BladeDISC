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


from __future__ import print_function
import os
import bazel_build
import torch

from setuptools import setup, find_packages
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext


builder = bazel_build.BazelBuild(
    torch.version.__version__,
    os.path.dirname(torch.__file__))
class TorchBladeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class TorchBladeBuild(build_ext):
    def run(self):
        # version.txt Would be package into C++ SDK by CPACK
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            builder.run(extdir=extdir, srcdir=ext.sourcedir, build_temp="build/temp")


class TestCommand(Command):
    user_options = []
    def initialize_options(self):
        # must override abstract method
        pass

    def finalize_options(self):
        # must override abstract method
        pass

    def run(self):
        builder.test()


setup(
    name='torch_disc',
    version='0.1',
    description='DISC backend implementation for Lazy tensors Core',
    url='https://github.com/alibaba/BladeDISC',
    author='DISC Dev Team',
    author_email='disc-dev@alibaba-inc.com',
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    ext_modules=[TorchBladeExtension("torch_disc._torch_disc")],
    cmdclass=dict(
        build_ext=TorchBladeBuild,
        test=TestCommand),
    package_data = {
        'torch_disc': [
            'lib/*.so*',
    ]},
    data_files=[]
)

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
import os
import re
import subprocess
import setuptools
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext

from version import __version__


def _get_device():
    # To get device info from bazel configuration.
    # Another option is to retrieve from TensorFlow API. But since we may build GPU 
    # package in a docker container with no GPU device mounted, tf.test.is_gpu_available
    # becomes not suitable. Reading from generated bazel configuration file to keep
    # consistent with C++ part.
    with open("./.bazelrc_gen", "r") as f:
        devices = []
        for line in f:
            line = line.strip()
            if 'build --config=cuda' == line:
                devices.append('gpu')
            elif 'build --config=cpu' == line:
                devices.append('cpu')
        assert (
            len(devices) == 1
        ), f"Multiple devices detected from .bazelrc_gen file: {devices}"
        return devices[0]


device = _get_device()

install_requires = [
    'numpy',
    'onnx>=1.6',
]

if device == 'gpu':
    install_requires.extend(['tf2onnx>=1.9.1'])


def _get_version(device):  # noqa: C901
    tf_version = ''
    try:
        import tensorflow.compat.v1 as tf

        tf_version = tf.__version__
    except ModuleNotFoundError:
        try:
            import tensorflow as tf

            tf_version = tf.__version__
        except ModuleNotFoundError:
            pass

    if device == 'gpu':
        if os.path.exists('/usr/local/cuda/version.txt'):
            with open('/usr/local/cuda/version.txt', 'r') as f:
                full_version_info = f.read()
                detailed_version = full_version_info.split(' ')[-1].split('.')
                version = detailed_version[0] + detailed_version[1]
                return '' if version == '100' else '+cu' + version
        # CUDA11 folder does not contain version.txt
        # so we use the ways defined in pytorch to get cuda version
        cmd = ["/usr/local/cuda/bin/nvcc", "--version"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
        out = out.decode("utf-8")
        if proc.returncode == 0:
            pattern = re.compile('release (.+),')
            release_version = pattern.findall(out)[0]
            major_minor = "".join(release_version.split(".")[:2])
            if len(tf_version) == 0:
                raise Exception("Failed to get tf version")
            else:
                return f'+cu{major_minor}.{tf_version}'
        raise Exception("Failed to get cuda version")
    else:
        if len(tf_version) != 0:
            return f'+{tf_version}'
        else:
            raise Exception("Failed to get tf version")


class CppBuild(build_ext):
    def run(self):
        assert len(self.extensions) == 1, "We've just a single extension."
        ext = self.extensions[0]
        print(f"[DEBUG] ext: {ext.name} - {ext.sourcedir}")
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        subprocess.check_call(
            "bazel build //src:_tf_blade.so", shell=True, executable="/bin/bash"
        )
        bazel_bin_dir = os.path.join(ext.sourcedir, "bazel-bin")
        for fpath in ['src/libtf_blade.so', 'src/_tf_blade.so']:
            fpath = os.path.join(bazel_bin_dir, fpath)
            fname = os.path.basename(fpath)
            assert os.path.exists(fpath), f"{fpath} not found!"
            link_name = os.path.join(extdir, fname)
            if os.path.exists(link_name):
                os.remove(link_name)
            os.symlink(fpath, link_name)
            print(f"Link: {link_name} -> {fpath}")


class CppTestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("[TODO] CppTestCommand run....")
        pass


class TfBladeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


setuptools.setup(
    name="tensorflow-blade-" + device,
    version=(__version__ + _get_version(device)),
    author="Alibaba PAI Team",
    description="TensorFlow-Blade is a general automatic inference optimization system.",
    packages=setuptools.find_packages(exclude=['src', 'src.*', 'tests']),
    install_requires=install_requires,
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.6',
    ext_modules=[TfBladeExtension("tf_blade._tf_blade")],
    cmdclass=dict(
        build_ext=CppBuild,
        cpp_test=CppTestCommand,
    ),
)

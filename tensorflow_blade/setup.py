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
import glob
import os
import re
import sys
import subprocess
import setuptools
from setuptools import Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

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


DEVICE = _get_device()

def get_install_requires():
    install_requires = ['numpy', 'onnx>=1.6']
    if DEVICE == 'gpu':
        install_requires.extend(['tf2onnx>=1.9.1'])
    return install_requires


def _get_version():  # noqa: C901
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

    if DEVICE == 'gpu':
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
        assert isinstance(self.extensions[0], TfBladeExtension)
        ext = self.extensions[0]
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # bazel build tf_blade extension.
        subprocess.check_call(
            "bazel build //src:_tf_blade.so", shell=True, executable="/bin/bash"
        )
        lib_pat = re.compile(r".+\.so(\.[0-9]+)?$")
        BIN_LIST = ['hie_serialize']

        # remove old links.
        for fname in os.listdir(extdir):
            if lib_pat.match(fname) or fname in BIN_LIST:
                full_path = os.path.join(extdir, fname)
                if os.path.islink(full_path):
                    os.remove(full_path)
                    print(f"Unlink native lib: {full_path}")

        # link native libraries.
        bazel_bin_dir = os.path.join(ext.sourcedir, "bazel-bin")
        for search_dir in ['src', os.path.join('src', 'internal')]:
            for fpath in glob.glob(os.path.join(bazel_bin_dir, search_dir, '*')):
                fname = os.path.basename(fpath)
                if lib_pat.match(fpath) or fname in BIN_LIST:
                    link_name = os.path.join(extdir, fname)
                    if os.path.exists(link_name):
                        os.remove(link_name)
                    os.symlink(fpath, link_name)
                    print(f"Link native lib: {fpath}")

        # link internal disc.
        import tensorflow as tf
        compiler_fname = 'tao_compiler_main'
        compiler_bin = os.path.join(
            os.path.dirname(tf.__file__), os.path.pardir, 'aicompiler', compiler_fname)
        compiler_bin = os.path.abspath(compiler_bin)
        if os.path.exists(compiler_bin):
            link_name = os.path.join(extdir, compiler_fname)
            if os.path.exists(link_name):
                os.remove(link_name)
            os.symlink(compiler_bin, link_name)

        # other extensions
        self.extensions.pop(0)
        super().run()


class CppTestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        filter = "-cpu" if DEVICE == "gpu" else "-gpu"
        subprocess.check_call(
            f"bazel test //src/... --build_tests_only --test_tag_filters={filter}",
            shell=True,
            executable="/bin/bash",
        )


class TfBladeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


ext_modules = [TfBladeExtension("tf_blade._tf_blade")]
exclude_py_mods = []
if 'develop' not in sys.argv and os.path.exists('tf_blade/internal'):
    cython_modules = cythonize(
            module_list=['tf_blade/internal/**/*.py'],
            compiler_directives={'language_level': 3},
            build_dir="build",
            nthreads=8
            )
    exclude_py_mods = [m.name for m in cython_modules]
    ext_modules.extend(cython_modules)


class PyBuild(build_py):
    """ Just to exclude cythonized .py files."""

    def find_modules(self):
        modules = super().find_modules()
        return [
            (pkg, mod, file,)
            for pkg, mod, file in modules
            if pkg + '.' + mod not in exclude_py_mods
        ]

    def find_package_modules(self, package: str, package_dir: str):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file,)
            for pkg, mod, file in modules
            if pkg + '.' + mod not in exclude_py_mods
        ]


setuptools.setup(
    name="tensorflow-blade-" + DEVICE,
    version=(__version__ + _get_version()),
    author="Alibaba PAI Team",
    description="TensorFlow-Blade is a general automatic inference optimization system.",
    packages=setuptools.find_packages(exclude=['src', 'src.*', 'tests', 'tests.*']),
    package_data={'tf_blade': ['py.typed']},
    install_requires=get_install_requires(),
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.6',
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CppBuild, cpp_test=CppTestCommand, build_py=PyBuild),
)

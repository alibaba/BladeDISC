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
import fnmatch
import os
import re
import subprocess
import sys
from itertools import chain
from typing import List

import setuptools
from setuptools.command.build_py import build_py

from version import __version__

SKIP_TRT = os.environ.get('SKIP_TRT').lower() == 'true'


def get_tf_blade_files() -> List[str]:
    """
    Just list all files under tf_blade and all .so files under other directoies. Bazel
    helps to put native files on correct place.
    """
    res = []
    for root, dirs, files in os.walk('tf_blade/'):
        root = root[len('tf_blade/') :]
        for fname in files:
            if '.so' in fname or '.so.' in fname:
                res.append(os.path.join(root, fname))
    return res


device = os.environ.get('PKG_DEVICE', 'gpu').lower()
if device is not None:
    if device not in ['cpu', 'gpu']:
        raise Exception("The device must be in choice of ['cpu', 'gpu']")

install_requires = [
    'numpy',
    'onnx>=1.6',
]

if device == 'gpu':
    install_requires.extend(['tf2onnx>=1.9.1'])


def get_version(device):  # noqa: C901
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
                return f'+cu{major_minor}_{tf_version}'
        raise Exception("Failed to get cuda version")
    else:
        if len(tf_version) != 0:
            return f'+{tf_version}'
        else:
            raise Exception("Failed to get tf version")

is_develop = 'develop' in sys.argv
print("IS DEVELOP MODE: ", is_develop)

packages = setuptools.find_packages(
    exclude=['src', 'src.*']
)

package_data = {
    "tf_blade": get_tf_blade_files(),
}

setuptools.setup(
    name="tensorflow-blade-" + device,
    version=(__version__ + get_version(device)),
    author="Alibaba PAI Team",
    # TODO(xiafei.qiuxf): need a public email address.
    # author_email="author@example.com",
    description="TensorFlow-Blade is a general automatic inference optimization system.",
    packages=packages,
    package_data=package_data,
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
    ext_modules=[],
    cmdclass={},
)

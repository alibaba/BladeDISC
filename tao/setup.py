#!/usr/bin/env python
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


# -*- coding: utf-8 -*-

import io
import os

import pkg_resources
from setuptools import find_packages, setup

import tensorflow as tf

# Package meta-data.
NAME_PREFIX = 'blade-disc'
DESCRIPTION = 'TensorFlow wrapper for Blade DISC compiler.'
URL = 'https://https://github.com/alibaba/BladeDISC'
EMAIL = 'tashuang.zk@alibaba-inc.com'
AUTHOR = 'Zhu Kai'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None


def detect_host_tf_version():
    for pkg in pkg_resources.working_set:
        if 'tensorflow-io' in pkg.project_name:
            continue
        if 'tensorflow' in pkg.project_name:
            return f"{pkg.project_name}=={pkg.version}"

# Format the Blade-DISC package name prefix on GPU or CPU:
# blade-disc-gpu-tf24 for tensorflow==2.4
# blade-disc-tf115 for tensorflow-gpu==1.15
def format_package_name():
    tf_short = "-tf{}".format("".join(tf.__version__.split(".")))
    gpu = "-gpu" if tf.test.is_gpu_available() else ""
    return "{}{}{}".format(NAME_PREFIX, gpu, tf_short)


# What packages are required for this module to be executed?
REQUIRED = [
    detect_host_tf_version()
]

SETUP_REQUIRED = [
    'pytest-runner'
]
TEST_REQUIRED = [
    'pytest',
]

# What packages are optional?
EXTRAS = {
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, 'python', 'blade_disc_tf', '_version.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=format_package_name(),
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages('python', exclude=('tests',)),
    install_requires=REQUIRED,
    setup_requires=SETUP_REQUIRED,
    extras_require=EXTRAS,
    license='Apache License 2.0',
    package_dir={'blade_disc_tf': 'python/blade_disc_tf'},
    package_data={'blade_disc_tf': ['libtao_ops.so', 'tao_compiler_main']},
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    zip_safe=False,
)

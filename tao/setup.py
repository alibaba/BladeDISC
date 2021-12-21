#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import re

import subprocess
from setuptools import find_packages, setup

# Package meta-data.
NAME_PREFIX = 'blade-disc'
DESCRIPTION = 'TensorFlow wrapper for Blade DISC compiler.'
URL = 'https://github.com/pai_disc/aicompiler'
EMAIL = 'tashuang.zk@alibaba-inc.com'
AUTHOR = 'Zhu Kai'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None


def cuda_version():
    return os.environ["CUDA_VERSION"]


def tensorflow_package():
    cuda = cuda_version()
    if not cuda:
        return "tensorflow==2.4"
    if cuda.startswith("10.0"):
        return "tensorflow-gpu==1.15"
    elif cuda.startswith("11.0"):
        return "tensorflow-gpu==2.4"
    else:
        raise NotImplementedError("should run blade_disc_tf on CUDA"
                                  "[10.0, 11.1] or CPU host")


# Format the Blade-DISC package name prefix on GPU or CPU:
# blade-disc-gpu-tf24 for tensorflow==2.4
# blade-disc-tf115 for tensorflow-gpu==1.15
def format_package_name():
    tf = tensorflow_package()
    tf_short = "-tf{}".format("".join(tf.split("==")[1].split(".")))
    gpu = "-gpu" if cuda_version() else ""
    return "{}{}{}".format(NAME_PREFIX, gpu, tf_short)


# What packages are required for this module to be executed?
REQUIRED = [
    tensorflow_package()
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Alibaba Inc.
# File              : setup.py
# Author            : fl237079 <fl237079@alibaba-inc.com>
# Date              : 2022-06-14
# Last Modified Date: 2022-06-14
# Last Modified By  : fl237079 <fl237079@alibaba-inc.com>


import setuptools
import shutil
import os, glob


setuptools.setup(
    name='disc_opt',
    version='2022.01.27',
    description='Optimization tools for TensorFlow Execution.',
    author='alibaba pai',
    packages=setuptools.find_packages(),
    package_data={
        "": ["libtao_ops.so", "tao_compiler_main", "libstdc++.so.6",
            "kernel_cache/*", "tvm/lib*.so", "ld.lld",
            "profile_cache/kernel_tune.json", "profile_cache/kernel_tune.log"]
    },
    entry_points={'console_scripts': 
     ['disc_kernel_gen=disc_opt.kernel_gen:execute', 
     ]},
    install_requires=[
        'numpy',
        'decorator',
        'attrs',
        'tornado',
        'psutil',
        'xgboost',
        'cloudpickle',
        'filelock',
    ],
)

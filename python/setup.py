#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Alibaba Inc.
# File              : setup.py
# Author            : Yue Wu <matthew.wy@alibaba-inc.com>
# Date              : 2022-03-24
# Last Modified Date: 2022-03-24
# Last Modified By  : Yue Wu <matthew.wy@alibaba-inc.com>


import setuptools
import shutil
import os, glob

# pkg_dir = os.path.dirname(os.path.abspath(__file__))
# tvm_home = os.environ.get("TVM_HOME", os.path.join(pkg_dir, "..", "..", "..", "tvm"))
# src = os.path.join(tvm_home, "python", "tvm")
# tvm_pkg = os.path.join(pkg_dir, "disc_dcu", "tvm")
# shutil.rmtree(tvm_pkg)
# shutil.copytree(src, tvm_pkg)
# for i in glob.glob(os.path.join(tvm_home, "build", "lib*.so")):
#     dst = os.path.join(tvm_pkg, os.path.basename(i))
#     shutil.copy(i, dst)


setuptools.setup(
    name='disc_dcu',
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
     ['disc_kernel_gen=disc_dcu.kernel_gen:execute', 
    # 'sr_mobile_convert=sr_mobile_converter.export:export_entry',
    # 'sr_kernel_check=sr_mobile_converter.check:check_entry',
    # 'sr_infer_check=sr_mobile_converter.debug:debug_entry',
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

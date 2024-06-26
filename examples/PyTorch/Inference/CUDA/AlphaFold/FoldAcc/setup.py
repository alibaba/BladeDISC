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

from torch.utils.cpp_extension import BuildExtension, CppExtension

from setuptools import find_packages, setup

requirements = ["torch", "scipy"]

cpp_extension = CppExtension(
    "foldacc_custom",
    ["foldacc/optimization/distributed/kernel/custom.cpp"],
)

setup(
    name="FoldAcc",
    version="0.1",
    author="blade",
    description="foldacc: an alphafold accleration framework.",
    packages=find_packages("."),
    install_requires=requirements,
    ext_modules= [cpp_extension],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
)
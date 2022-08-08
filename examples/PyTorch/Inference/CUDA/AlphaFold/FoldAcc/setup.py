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
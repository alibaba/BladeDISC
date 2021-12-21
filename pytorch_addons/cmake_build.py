import argparse
import os
import sys
import subprocess
import platform
import shutil
import re
from distutils.spawn import find_executable
import release_version

cwd = os.path.dirname(os.path.abspath(__file__))


def check_env_flag(name, default=None):
    env_val = os.getenv(name, default)
    return (
        None if env_val is None else env_val.upper() in ["ON", "1", "YES", "TRUE", "Y"]
    )


def get_fullpath_or_create(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.abspath(dir_path)


class CMakeBuild:
    def __init__(
        self,
        torch_dir,
        torch_version,
        cuda_version=None,
        torch_git_version=None,
        cxx11_abi=False,
    ):
        self.__serialization_version__ = "0.0.3"
        self.torch_dir = torch_dir
        self.cuda_version = cuda_version
        self.torch_version = torch_version
        self.git_version = self.get_git_version()
        self.torch_git_version = torch_git_version
        self.GLIBCXX_USE_CXX11_ABI = cxx11_abi
        # NB: Bump up because of MLIR Engine serialization changes
        self.is_debug = check_env_flag("DEBUG", "OFF")
        self.cuda_available = check_env_flag(
            "TORCH_ADDONS_BUILD_WITH_CUDA_SUPPORT", "ON"
        )
        self.version = self.get_version()

    def get_git_version(self):
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
                .decode("ascii")
                .strip()
            )
        except Exception as e:
            print("\t", e)
            sha = "unknown"
        return sha

    def get_version(self):
        version = release_version.__version__
        if version == "0.0.0":
            # this is develop version
            version += ".dev0"

        if not self.cuda_version == "10.0":
            torch_ver = self.torch_version.split("+")[0]
            if self.cuda_version:
                cuda_ver = self.cuda_version.replace(".", "")
                torch_ver += f".cu{cuda_ver}"

            version += f"+{torch_ver}"
        return version

    def write_version_file(self, version_path):
        with open(version_path, "w") as f:
            f.write("__version__ = {}\n".format(repr(self.version)))
            f.write(
                "__serialization_version__ = {}\n".format(
                    repr(self.__serialization_version__)
                )
            )
            f.write("debug = {}\n".format(repr(self.is_debug)))
            f.write("cuda = {}\n".format(repr(self.cuda_version)))
            f.write("cuda_available = {}\n".format(repr(self.cuda_available)))
            f.write("git_version = {}\n".format(repr(self.git_version)))
            f.write("torch_version = {}\n".format(repr(self.torch_version)))
            f.write("torch_git_version = {}\n".format(repr(self.torch_git_version)))
            f.write(
                "GLIBCXX_USE_CXX11_ABI = {}\n".format(repr(self.GLIBCXX_USE_CXX11_ABI))
            )

        with open(version_path, "r") as f:
            print("".join(f.readlines()))

    def run(self, extdir=None, srcdir=None, build_temp=None):
        extdir = get_fullpath_or_create(extdir or "build/temp")
        srcdir = get_fullpath_or_create(srcdir or os.path.dirname(os.path.abspath(__file__)))
        build_temp = get_fullpath_or_create(build_temp or extdir)
        build_temp = build_temp or extdir

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the project")

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        self.build_extension(extdir, srcdir, build_temp)
        try:
            self.patchelf_fix_sonames(extdir)
        except Exception:
            pass

    def build_extension(self, extdir, srcdir, build_temp):
        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
        torch_major_version, torch_minor_version = self.torch_version.split(".")[:2]
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPYTHON_VERSION=" + py_version,
            # PyTorch cmake args
            "-DPYTORCH_VERSION_STRING=" + self.torch_version,
            "-DPYTORCH_MAJOR_VERSION=" + torch_major_version,
            "-DPYTORCH_MINOR_VERSION=" + torch_minor_version,
            "-DPYTORCH_DIR={}".format(self.torch_dir),
            "-DTORCH_ADDONS_USE_CXX11_ABI={}".format(self.GLIBCXX_USE_CXX11_ABI),
            "-DTORCH_ADDONS_CUDA_VERSION={}".format(self.cuda_version),
            "-DTORCH_ADDONS_BUILD_WITH_CUDA_SUPPORT={}".format(self.cuda_available),
        ]
        ccache = os.environ.get("CCACHE", None)
        executable = (
            ccache is not None and os.path.exists(ccache) and os.access(ccache, os.X_OK)
        )
        if executable:
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=%s" % ccache,
                "-DCMAKE_CXX_COMPILER_LAUNCHER=%s" % ccache,
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=%s" % ccache,
            ]

        # Use ninja if it's available
        if find_executable("ninja"):
            cmake_args.append("-GNinja")

        cfg = "Debug" if self.is_debug else "Release"
        build_args = ["--config", cfg, "--target", "package"]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j%d" % os.cpu_count()]

        def get_cmake_env_arg(env_var, default_val):
            env_val = check_env_flag(env_var, default_val)
            return [] if env_val is None else [f"-D{env_var}={env_val}"]

        cmake_args += get_cmake_env_arg("TORCH_ADDONS_BUILD_MLIR_SUPPORT", None)
        cmake_args += get_cmake_env_arg("TORCH_ADDONS_BUILD_PYTHON_SUPPORT", None)
        cmake_args += get_cmake_env_arg("TORCH_ADDONS_PLATFORM_ALIBABA", None)

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.version
        )
        subprocess.check_call(["cmake", srcdir] + cmake_args, cwd=build_temp, env=env)
        shutil.copyfile(
            os.path.join(build_temp, "compile_commands.json"),
            os.path.join(srcdir, "compile_commands.json"),
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, env=env
        )
        with open("cpp_test.sh", "w") as out_f:
            out_f.write('cd {} && ctest "$@"\n'.format(build_temp))

    def patchelf_fix_sonames(self, extdir):
        if self.cuda_version is None:
            return
        if check_env_flag("TORCH_ADDONS_DISABLE_PATCHELF_CUDA_SONAMES"):
            return

        torch_libs_dir = os.path.join(self.torch_dir, "lib")
        cuda_ver = self.cuda_version
        deps_lib_patch_regex = {
            f"libnvrtc.so.{cuda_ver}": f"libnvrtc(-[0-9a-fA-F]+)?.so.{cuda_ver}",
            f"libcudart.so.{cuda_ver}": f"libcudart(-[0-9a-fA-F]+)?.so.{cuda_ver}",
            "libnvToolsExt.so.1": "libnvToolsExt(-[0-9a-fA-F]+)?.so.1",
        }
        libs = [os.path.basename(lib) for lib in os.listdir(torch_libs_dir)]
        # search and save patchelf sonames from torch libraries if exists
        deps_patch_lib = dict()
        for deps_lib, patch_regex in deps_lib_patch_regex.items():
            regex = re.compile(patch_regex)
            for lib in libs:
                if regex.match(lib):
                    deps_patch_lib[deps_lib] = lib
                    shutil.copy(
                        os.path.join(torch_libs_dir, lib), os.path.join(extdir, lib)
                    )

        torch_addons_libs = [
            os.path.join(extdir, lib)
            for lib in os.listdir(extdir)
            if "libtorch_addons" in lib
        ]
        # use patchelf to replace needed sonames
        for origin_soname, patch_soname in deps_patch_lib.items():
            for lib in torch_addons_libs:
                subprocess.check_output(
                    ["patchelf", "--replace-needed", origin_soname, patch_soname, lib]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMake build TorchAddons")
    parser.add_argument(
        "--torch_version", type=str, required=True, help="The version of torch"
    )
    parser.add_argument(
        "--torch_dir", type=str, required=True, help="The directory where torch located"
    )
    parser.add_argument(
        "--cuda_version", type=str, default=None, help="The version of cuda toolkit"
    )
    parser.add_argument("--cxx11", action="store_true", help="Use c++ cxx11 abi")

    args = parser.parse_args()

    build = CMakeBuild(
        args.torch_dir, args.torch_version, args.cuda_version, cxx11_abi=args.cxx11
    )
    build.write_version_file(os.path.join(cwd, "version.txt"))
    build.run()

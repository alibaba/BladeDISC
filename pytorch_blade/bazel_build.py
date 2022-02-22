import argparse
import os
import errno
import subprocess
import sys
import venv

from torch_blade_build import TorchBladeBuild, get_fullpath_or_create

cwd = os.path.dirname(os.path.abspath(__file__))


def _symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

class BazelBuild(TorchBladeBuild):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_suite = "//src:torch_blade_gtests"
        self.targets = [
            "@org_tensorflow//tensorflow/compiler/mlir/disc:disc_compiler_main",
            "//src:_torch_blade.so",
            self.test_suite,
        ]
        self.torch_lib_dir = os.path.join(self.torch_dir, 'lib')
        torch_major_version, torch_minor_version = self.torch_version.split(".")[:2]
        self.extra_opts = [
            "--copt=-DPYTORCH_VERSION_STRING=" + self.torch_version,
            "--copt=-DPYTORCH_MAJOR_VERSION=" + torch_major_version,
            "--copt=-DPYTORCH_MINOR_VERSION=" + torch_minor_version,
            "--copt=-DTORCH_BLADE_USE_CXX11_ABI={}".format(self.GLIBCXX_USE_CXX11_ABI),
            "--copt=-DTORCH_BLADE_CUDA_VERSION={}".format(self.cuda_version),
            "--action_env PYTHON_BIN_PATH={}".format(sys.executable),
            "--action_env TORCH_BLADE_TORCH_INSTALL_PATH={}".format(self.torch_dir),
            # Workaroud issue: https://github.com/bazelbuild/bazel/issues/10327
            "--action_env BAZEL_LINKLIBS=-lstdc++"
        ]

        self.configs = []
        if self.is_debug:
            self.configs.append("--config=dbg")

        if self.cuda_available:
            self.configs.append("--config=torch_disc_cuda")
        else:
            self.configs += ["--config=torch_disc_cpu"]

        self.shell_setting = "set -e; set -o pipefail; "
        # Workaround: this venv ensure that $(/usr/bin/env python) is evaluated to python3
        venv.create("bazel_pyenv")
        self.build_cmd = "source bazel_pyenv/bin/activate; bazel build --experimental_repo_remote_exec"
        self.test_cmd = "source bazel_pyenv/bin/activate; bazel test --experimental_repo_remote_exec"

    def run(self, extdir=None, srcdir=None, build_temp=None):
        srcdir = get_fullpath_or_create(
            srcdir or os.path.dirname(os.path.abspath(__file__))
        )
        extdir = get_fullpath_or_create(extdir or "build/temp")
        bazel_bin_dir = os.path.join(srcdir, "bazel-bin/")

        env = os.environ.copy()
        ld_library_path = ":".join([self.torch_lib_dir, env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path

        bazel_cmd = " ".join(
            [self.shell_setting, self.build_cmd]
            + self.extra_opts
            + self.configs
            + self.targets
        )
        subprocess.check_call(
            bazel_cmd, shell=True, env=env, executable="/bin/bash"
        )

        ext_so_fpath = "src/_torch_blade.so"
        ral_so_fpath = "external/org_tensorflow/tensorflow/compiler/mlir/xla/ral/libral_base_context.so"
        disc_bin_fpath = (
            "external/org_tensorflow/tensorflow/compiler/mlir/disc/disc_compiler_main"
        )

        for fpath in [ext_so_fpath, ral_so_fpath, disc_bin_fpath]:
            fpath = os.path.realpath(os.path.join(bazel_bin_dir, fpath))
            fname = os.path.basename(fpath)
            _symlink_force(fpath, os.path.join(extdir, fname))

    def test(self):
        env = os.environ.copy()
        ld_library_path = ":".join([self.torch_lib_dir, env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path

        test_cmd = " ".join(
            [self.shell_setting, self.test_cmd]
            + self.extra_opts
            + self.configs
            + [self.test_suite]
        )
        subprocess.check_call(test_cmd, shell=True, env=env, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bazel build TorchBlade")
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

    build = BazelBuild(
        args.torch_dir, args.torch_version, args.cuda_version, cxx11_abi=args.cxx11
    )
    build.write_version_file(os.path.join(cwd, "version.txt"))
    srcdir = os.path.dirname(os.path.abspath(__file__))
    build.run(extdir=os.path.join(srcdir, "torch_blade"))

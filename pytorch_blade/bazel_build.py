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


def _gen_torch_bazel_workspace(torch_dir, srcdir):
    torch_bzl_tmpl = f"""
def torch_env_workspace():
    native.new_local_repository(
        name = "local_org_torch",
        path = "{torch_dir}",
        build_file = "@//bazel/torch:BUILD",
    )

def torch_pybind11_dir():
    return "{torch_dir}/include/pybind11"
"""
    with open(f"{srcdir}/bazel/torch/.torch_env_workspace.bzl", "w") as f:
        f.write(torch_bzl_tmpl)


class BazelBuild(TorchBladeBuild):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_suite = "//src:torch_blade_gtests"
        self.targets = [
            "@org_tensorflow//tensorflow/compiler/mlir/disc:disc_compiler_main",
            "//src:_torch_blade.so",
            self.test_suite,
        ]

        self.shell_setting = "set -e; set -o pipefail; "
        self.configs = []
        self.copts = []

        if self.is_debug:
            self.configs.append("--config=dbg")

        if self.cuda_available:
            self.configs.append("--config=torch_disc_cuda")
        else:
            self.configs += ["--config=torch_disc_cpu"]

        venv.create("bazel_pyenv")
        self.build_cmd = "source bazel_pyenv/bin/activate; bazel build --experimental_repo_remote_exec -s"
        self.test_cmd = "source bazel_pyenv/bin/activate; bazel test --experimental_repo_remote_exec"

    def run(self, extdir=None, srcdir=None, build_temp=None):
        srcdir = get_fullpath_or_create(
            srcdir or os.path.dirname(os.path.abspath(__file__))
        )
        extdir = get_fullpath_or_create(extdir or "build/temp")
        bazel_bin_dir = os.path.join(srcdir, "bazel-bin/")

        _gen_torch_bazel_workspace(self.torch_dir, srcdir)
        env = os.environ.copy()
        env["PYTHON_BIN_PATH"] = sys.executable

        bazel_cmd = " ".join(
            [self.shell_setting, self.build_cmd]
            + self.copts
            + self.configs
            + self.targets
        )
        subprocess.check_call(
            bazel_cmd, shell=True, env=env
        )  # executable=sys.executable)

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
        env["PYTHON_BIN_PATH"] = sys.executable

        test_cmd = " ".join(
            [self.shell_setting, self.test_cmd]
            + self.copts
            + self.configs
            + [self.test_suite]
        )
        subprocess.check_call(test_cmd, shell=True, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMake build TorchBlade")
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

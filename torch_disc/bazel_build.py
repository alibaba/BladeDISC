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

import argparse
import os
import errno
import subprocess
import sys

cwd = os.path.dirname(os.path.abspath(__file__))

def get_fullpath_or_create(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.abspath(dir_path)

def _make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

def _symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

class BazelBuild():
    def __init__(self, torch_version, torch_dir):
        self.torch_dir = torch_dir
        self.torch_lib_dir = os.path.join(self.torch_dir, 'lib')
        self.torch_version = torch_version
        self.targets = [
            "//torch_disc:_torch_disc.so",
        ]
        torch_major_version, torch_minor_version = self.torch_version.split(".")[:2]
        self.extra_opts = [
            "--copt=-DPYTORCH_MAJOR_VERSION={}".format(torch_major_version),
            "--copt=-DPYTORCH_MINOR_VERSION={}".format(torch_minor_version),
            "--action_env TORCH_BLADE_TORCH_INSTALL_PATH={}".format(torch_dir),
            # Workaroud issue: https://github.com/bazelbuild/bazel/issues/10327
            "--action_env BAZEL_LINKLIBS=-lstdc++"
        ]

        self.shell_setting = "set -e; set -o pipefail; "
        self.build_cmd = "bazel build"
        self.ci_flag = "--noshow_loading_progress --show_progress_rate_limit=600"

    def fix_generated_code(self):
        cmd = [os.path.join("scripts", "pytorch_patch.sh")]
        if subprocess.call(cmd) != 0:
            print(
                'Failed to correct ATEN bindins head files: {}'.format(cmd),
                file=sys.stderr)
            sys.exit(1)

    def run(self, extdir=None, srcdir=None, build_temp=None):
        self.fix_generated_code()
        srcdir = get_fullpath_or_create(
            srcdir or os.path.dirname(os.path.abspath(__file__))
        )
        extdir = get_fullpath_or_create(extdir or "build/temp")

        env = os.environ.copy()
        ld_library_path = ":".join([env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path

        bazel_cmd = " ".join(
            [self.shell_setting, self.build_cmd]
            + self.extra_opts
        )
        if os.getenv("GITHUB_ACTIONS", ""):
            bazel_cmd += "--noshow_loading_progress --show_progress_rate_limit=600"

        with open("debug_bazel.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("export LD_LIBRARY_PATH={}\n".format(ld_library_path))
            f.write("export GCC_HOST_COMPILER_PATH={}\n".format(env.get("GCC_HOST_COMPILER_PATH", "")))
            f.write(bazel_cmd + " $@")
        _make_executable("debug_bazel.sh")

        bazel_cmd = " ".join([bazel_cmd] + self.targets)

        subprocess.check_call(
            bazel_cmd, shell=True, env=env, executable="/bin/bash"
        )

    def test(self):
        env = os.environ.copy()
        ld_library_path = ":".join([self.torch_lib_dir, env.get("LD_LIBRARY_PATH", "")])
        env["LD_LIBRARY_PATH"] = ld_library_path
        test_suite = [
            "//torch_disc:torch_disc_test_suit",
        ]
        test_cmd = "bazel test"

        test_cmd = " ".join(
            [self.shell_setting, test_cmd]
            + self.extra_opts
            + test_suite
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

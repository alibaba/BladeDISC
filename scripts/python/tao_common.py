# !/usr/bin/env python3
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

import subprocess
import os
import shutil
import re
from contextlib import contextmanager
from build_common import logger

GCC_48_BIN_PATH = '/usr/bin'
GCC_48_LIB_PATH = '/usr/lib64'
GCC_49_BIN_PATH = '/usr/local/alicpp/built/gcc-4.9.2/gcc-4.9.2/bin'
GCC_49_LIB_PATH = '/usr/local/alicpp/built/gcc-4.9.2/gcc-4.9.2/lib64'
GCC_53_BIN_PATH = '/usr/local/gcc-5.3.0/bin'
GCC_53_LIB_PATH = '/usr/local/gcc-5.3.0/lib64'
GCC_65_BIN_PATH = '/usr/local/alicpp/built/gcc-6.5.1/gcc-6.5.1/bin'
GCC_65_LIB_PATH = '/usr/local/alicpp/built/gcc-6.5.1/gcc-6.5.1/lib64'
GCC_73_BIN_PATH = '/opt/rh/devtoolset-7/root/usr/bin'
GCC_73_LIB_PATH = '/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7'
DEFAULT_BIN_PATH = '/usr/bin'
DEFAULT_LIB_PATH = '/usr/lib64'

# GCC version -> (bin_path, lib_path)
VALID_GCC = {
    '4.8': (GCC_48_BIN_PATH, GCC_48_LIB_PATH),
    '4.9': (GCC_49_BIN_PATH, GCC_49_LIB_PATH),
    '5.3': (GCC_53_BIN_PATH, GCC_53_LIB_PATH),
    '6.5': (GCC_65_BIN_PATH, GCC_65_LIB_PATH),
    '7.3': (GCC_73_BIN_PATH, GCC_73_LIB_PATH),
    'default': (DEFAULT_BIN_PATH, DEFAULT_LIB_PATH)
}

VALID_CUDA = ['9.0', '10.0', '10.1', '11.0']
VALID_CUDNN = {
    '9.0': ['7.3.1.20', '7.2.1.38'],
    '10.0': ['7.5.0.56', '7.6.3.30', '7.6.4.38', '7.6.5.32'],
    '10.1': ['7.6.4.38'],
    '11.0': ['8.0.5.39']
}

ENV_VAR_TMP_GCC = 'TAO_TMP_GCC'


def execute(cmd):
    """Execute a shell command, exception raised on failure."""
    shell_setting = "set -e; set -o pipefail; "
    gcc_info = os.environ.get(ENV_VAR_TMP_GCC) or "default"
    logger.info(
        "Execute shell command: `" + cmd + "`, cwd: " + os.getcwd() + ", gcc: " + gcc_info)
    subprocess.check_call(shell_setting + cmd, shell=True, executable='/bin/bash')


def git_branch():
    """Get current git branch."""
    br = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True)
    return br.strip()


def git_head():
    """Get current git HEAD commit."""
    head = subprocess.check_output('git rev-parse --verify HEAD', shell=True)
    return head.strip()


def detect_cuda_version():
    """
    return a tuple with major and minor version
    """
    nvcc_out = subprocess.check_output('nvcc --version', shell=True)
    if not nvcc_out:
        return None
    m = re.search('release ([0-9]+)\.([0-9])', str(nvcc_out))
    if not m:
        return None
    return (m.group(1), m.group(2))


def get_tf_gpu_version():
    """
    Get proper value of env var TF_GPU_VERSION, which is used to choose env.conf file:
    ${TF_PLATFORMS_DIR}/${TF_PLATFORM}/${TF_DEVICE}/env.conf.${TF_GPU_VERSION}"
    """
    vers = detect_cuda_version()
    if len(vers) == 2 and vers[1] != "0":
        # Special case for 10.1
        return "cuda{}_{}".format(vers[0], vers[1])
    else:
        return "cuda{}".format(vers[0])

@contextmanager
def default_env(var, default_val):
    """
    Use default envrion variable value if not set
    """
    is_set = var in os.environ
    try:
        if not is_set:
            os.environ[var] = default_val
        yield
    finally:
        if not is_set:
            os.environ.pop(var)

@contextmanager
def gcc_env(gcc_version):
    """
    Change the PATH and LD_LIBRARY_PATH to given GCC environment, these env
    vars will be restored when it's done.
    """
    def append_env_var(name, new_part):
        saved = os.environ.get(name)
        os.environ[name] = new_part + ":" + saved if saved else new_part
        return saved

    def restore_env_var(name, saved):
        if saved:
            os.environ[name] = saved
        else:
            del os.environ[name]

    gcc_compiler_configured = 'GCC_HOST_COMPILER_PATH' in os.environ
    os.environ[ENV_VAR_TMP_GCC] = gcc_version
    bin_path, lib_path = VALID_GCC[gcc_version]

    should_set_gcc = not (gcc_compiler_configured or bin_path is None or lib_path is None)
    if should_set_gcc:
        saved_path = append_env_var("PATH", bin_path)
        saved_host_gcc = append_env_var("GCC_HOST_COMPILER_PATH", bin_path + "/gcc")
        saved_ld_path = append_env_var("LD_LIBRARY_PATH", lib_path)
    try:
        yield
    finally:
        del os.environ[ENV_VAR_TMP_GCC]
        if should_set_gcc:
            restore_env_var("PATH", saved_path)
            restore_env_var("LD_LIBRARY_PATH", saved_ld_path)
            restore_env_var("GCC_HOST_COMPILER_PATH", saved_host_gcc)


def read_bool_from_env(name, default_val):
    value = os.getenv(name, None)
    if value is None:
        return default_val
    return value.lower() in ["true", "1"]


def overwrite_file(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def running_on_ci():
    """
    Return true if the building job is running on CI host.
    """
    if os.getenv("GITHUB_WORKFLOW"):
        return True
    return False


def ci_build_flag():
    if running_on_ci():
        return " --noshow_loading_progress --show_progress_rate_limit=600"
    return ""


def remote_cache_token():
    """
    Return a remote cache token if exists
    """
    fn = os.path.expanduser("~/.cache/remote_cache_token")
    if os.path.exists(fn):
        with open(fn) as f:
            return str(f.read()).strip()
    return None

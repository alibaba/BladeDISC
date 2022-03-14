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

# type: ignore
import subprocess
import logging
import sys
import os
import shlex
import shutil
import json
import re
from subprocess import Popen, PIPE, STDOUT
from contextlib import contextmanager

PY_VER = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

ENV_VAR_TMP_GCC = "BLADE_TMP_GCC"


def __create_logger():
    """Create a logger with color."""
    # The background is set with 40 plus the number of the color, and the foreground with 30
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": YELLOW,
        "ERROR": RED,
    }

    class ColoredFormatter(logging.Formatter):
        def __init__(self, msg, use_color=False):
            logging.Formatter.__init__(self, msg)
            self.use_color = use_color

        def format(self, record):
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = (
                    COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                )
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    class ColoredLogger(logging.Logger):
        FORMAT = "{}%(asctime)s{} %(levelname)19s %(message)s".format(
            BOLD_SEQ, RESET_SEQ
        )

        def __init__(self, name):
            logging.Logger.__init__(self, name, logging.DEBUG)
            color_formatter = ColoredFormatter(
                self.FORMAT, use_color=sys.stdout.isatty() and sys.stderr.isatty()
            )
            console = logging.StreamHandler()
            console.setFormatter(color_formatter)
            self.addHandler(console)
            return

    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger("blade_ci")
    logger.setLevel(logging.INFO)
    return logger


logger = __create_logger()


def execute(cmd, silent_fail=False):
    """Execute a shell command, exception raised on failure."""
    shell_setting = "set -e; set -o pipefail; "
    logger.info(
        "Execute shell command: `{}`, cwd: {}, silent_fail: {}".format(
            cmd, os.getcwd(), silent_fail
        )
    )
    try:
        subprocess.check_call(shell_setting + cmd, shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        if not silent_fail:
            raise e
        else:
            logger.warning(
                "Command execute failed, exit code: {}, cmd: {}".format(
                    e.returncode, e.cmd
                )
            )


def safe_run(cmd, shell=False, verbose=True):
    assert isinstance(cmd, str) or isinstance(cmd, unicode)
    if shell:
        popen = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=shell)
    else:
        args = shlex.split(cmd)
        popen = Popen(args, stdout=PIPE, stderr=STDOUT, shell=shell)
    # wait until subprocess terminated
    # stdout, stderr = popen.communicate()
    stdout = ""
    for line in iter(popen.stdout.readline, b""):
        clean_line = line.strip().decode("utf-8")
        if verbose:
            logger.info(clean_line)
        stdout += "{}\n".format(clean_line)
    if stdout and "error" in stdout.lower():
        logger.info(
            'Running "{}" with shell mode {}'.format(cmd, "ON" if shell else "OFF")
        )
        raise AssertionError("{} failed!".format(cmd))
    return stdout


def git_branch():
    """Get current git branch."""
    br = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
    return br.strip()


def git_head():
    """Get current git HEAD commit."""
    head = subprocess.check_output("git rev-parse --verify HEAD", shell=True)
    return head.strip()


def deduce_cuda_info():
    """Deduce cuda major and minor version and cuda directory."""

    def _deduce_from_version_file(cuda_home):
        version_file = os.path.join(cuda_home, "version.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                matched = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", f.read())
                if len(matched) == 1:
                    # return major and minor only.
                    return ".".join(matched[0].split(".")[0:2])
        version_file = os.path.join(cuda_home, "version.json")
        if os.path.exists(version_file):
            with open(version_file) as f:
                data = json.loads(f.read())
                parts = data['cuda']['version'].split(".")
                return parts[0] + "." + parts[1]
        return None

    def _deduce_from_nvcc():
        out = safe_run("nvcc --version", shell=True, verbose=False)
        patt = re.compile(r"release ([0-9]+\.[0-9]+)", re.M)
        found = patt.findall(out)
        if len(found) == 1:
            nvcc = which("nvcc")
            cuda_home = os.path.join(os.path.dirname(nvcc), os.path.pardir)
            return found[0], os.path.abspath(cuda_home)
        else:
            return None, None

    cuda_home = os.environ.get("BLADE_CUDA_HOME", None)
    if cuda_home:
        ver = _deduce_from_version_file(cuda_home)
        if ver is not None:
            return ver, cuda_home
        else:
            raise Exception(
                f"Failed to deduce cuda version from BLADE_CUDA_HOME: {cuda_home}"
            )

    ver = _deduce_from_version_file("/usr/local/cuda")
    if ver is not None:
        return ver, "/usr/local/cuda"

    all_cuda = [
        os.path.join("/usr/local", d)
        for d in os.listdir("/usr/local")
        if d.startswith("cuda-")
    ]
    assert (
        len(all_cuda) == 1
    ), "Mutiple cuda installed, but none linked to `/usr/local/cuda`."
    ver = _deduce_from_version_file(all_cuda[0])
    if ver is not None:
        return ver, all_cuda[0]

    ver, cuda_home = _deduce_from_nvcc()
    if ver is not None:
        return ver, cuda_home
    raise Exception("Failed to deduce cuda version from local installation.")


def get_cudnn_version(cuda_home):
    serched = []
    for hdr in ["cudnn.h", "cudnn_version.h"]:
        fname = os.path.join(cuda_home, "include", hdr)
        serched.append(fname)
        if not os.path.exists(fname):
            fname = os.path.join("/usr/include", hdr)
        with open(fname, "r") as f:
            major, minor, patch = None, None, None
            for line in f.readlines():
                line = line.strip()
                if "#define CUDNN_MAJOR" in line:
                    major = line.split(" ")[2]
                elif "#define CUDNN_MINOR" in line:
                    minor = line.split(" ")[2]
                elif "#define CUDNN_PATCHLEVEL" in line:
                    patch = line.split(" ")[2]
            if None not in [major, minor, patch]:
                return ".".join([major, minor, patch])
    raise Exception(f"Failed to decuce cuDNN version after searching: {fname}")


def get_trt_version(trt_home):
    hdr = os.path.join(trt_home, "include", "NvInferVersion.h")
    with open(hdr, "r") as f:
        major, minor, patch = None, None, None
        for line in f.readlines():
            line = line.strip()
            if "#define NV_TENSORRT_SONAME_MAJOR" in line:
                major = line.split(" ")[2]
            elif "#define NV_TENSORRT_SONAME_MINOR" in line:
                minor = line.split(" ")[2]
            elif "#define NV_TENSORRT_SONAME_PATCH" in line:
                patch = line.split(" ")[2]
        if None in [major, minor, patch]:
            raise Exception(f"Failed to decuce TensorRT version from: {hdr}")
        return ".".join([major, minor, patch])


def deduce_cuda_major_version():
    """Deduce cuda major version from env var or disk directory name."""
    return deduce_cuda_info()[0].split(".")[0]


def get_nv_driver_version():
    out = safe_run(
        'nvidia-smi --query | grep "Driver Version" | cut -d ":" -f 2 | xargs',
        shell=True,
        verbose=False,
    )
    nv_driver_version = out.strip("\n")
    return nv_driver_version


def get_gpu_id():
    gpu_id = (
        os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else 0
    )
    return gpu_id


def ensure_empty_dir(dir, clear_hidden=True):
    """
    Make sure the given directory exists and is empty.
    This function will create an empty directory if the directory doesn't exits,
    or it will clean all content under the directory. Hidden files and sub
    direcotries will be deleted if clear_hidden is True.
    """
    if not os.path.exists(dir):
        logger.info("make dir: " + dir)
        os.makedirs(dir)
        return
    logger.info("clear dir: {}, clear hidden files: {}".format(dir, clear_hidden))
    for filename in os.listdir(dir):
        if clear_hidden or not filename.startswith("."):
            file_path = os.path.join(dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            else:
                shutil.rmtree(file_path, ignore_errors=True)


def which(cmd):
    """ Same as `which` command of bash.
    """
    from distutils.spawn import find_executable

    found = find_executable(cmd)
    if not found:
        raise Exception("failed to find command: " + cmd)
    return found


@contextmanager
def cwd(path):
    """
    Change the current working directory to `path` to do somthing and then
    recover the current cwd when it's done.
    """
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

def get_site_packages_dir(venv_dir):
    venv_dir = os.path.abspath(venv_dir)
    python_bin = os.path.join(venv_dir, 'bin', 'python')
    output = subprocess.check_output(
        '{} -c "import sys; print(sys.path)"'.format(python_bin), shell=True
    ).decode()
    for path in output.strip().strip('[]').split(','):
        path = path.strip(" '\"")
        if path.endswith("site-packages") and path.startswith(venv_dir):
            return path
    raise Exception("site package path not exists")

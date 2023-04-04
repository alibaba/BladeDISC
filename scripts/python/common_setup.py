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
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from subprocess import PIPE, STDOUT, Popen


class StageTiming:
    def __init__(self):
        self.durs = []
        self.log_file = "/tmp/tao_build.log"

    def report(self):
        if len(self.durs) == 0:
            return
        lines = []
        # load previous info from file
        if os.path.exists(self.log_file):
            lines += open(self.log_file, "r").read().splitlines()
        for name, sec, ts in self.durs:
            lines.append("{}: {} - {:.2f} minutes".format(ts, name, sec * 1.0 / 60))
        # logger.info("****** Stage timing report: ******\n{}".format("\n".join(lines)))
        # logger.info("**********************************")
        # save to file
        with open(self.log_file, "w") as of:
            of.write("\n".join(lines))

    def append(self, name, secs):
        self.durs.append((name, secs, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


stage_time = StageTiming()


def time_stage(incl_args=[], incl_kwargs=[]):
    def time_stage_impl(entry):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                ret = entry(*args, **kwargs)
            except Exception:
                logger.exception("{} failed on exception".format(entry.__name__))
                raise Exception("run error")
            finally:
                end = time.time()
                name = entry.__name__
                if len(incl_args) > 0 or len(incl_kwargs) > 0:
                    name += "("
                    for idx in incl_args:
                        name += args[idx] + ","
                    for k in incl_kwargs:
                        name += kwargs[k] + ","
                    name = name[:-1] + ")"
                stage_time.append(name, end - start)
            return ret

        return wrapper

    return time_stage_impl


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_source_root_dir():
    root = os.path.join(script_dir(), os.pardir, os.pardir)
    return os.path.abspath(root)


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
    logger = logging.getLogger("tao_ci")
    logger.setLevel(logging.INFO)
    return logger


logger = __create_logger()


def execute(cmd):
    """Execute a shell command, exception raised on failure."""
    shell_setting = "set -e; set -o pipefail; "
    logger.info("Execute shell command: `" + cmd + "`, cwd: " + os.getcwd())
    subprocess.check_call(shell_setting + cmd, shell=True, executable="/bin/bash")


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
    """Same as `which` command of bash."""
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
    else:
        if "TF_REMOTE_CACHE" in os.environ:
            token = os.getenv("TF_REMOTE_CACHE")
            return token
    return None


def symlink_dir(srcdir, dstdir, excludes=None):
    """
    Recursively symlink all files and directories under `srcdir` to `dstdir`.
    """
    for f in os.listdir(srcdir):
        src_f = os.path.join(srcdir, f)
        dst_f = os.path.join(dstdir, f)
        if excludes and f in excludes:
            print(f"Exclude: {src_f}")
            continue

        if os.path.exists(dst_f) and os.path.samefile(src_f, dst_f):
            continue
        elif not os.path.exists(dst_f):
            cmd = ['ln', '-sfL', src_f, dst_f]
            print(" ".join(cmd))
            subprocess.check_output(cmd)
        elif os.path.isdir(src_f):
            symlink_dir(src_f, dst_f, excludes)
        else:
            cmd = ['ln', '-sfL', src_f, dst_f]
            print(" ".join(cmd))
            subprocess.check_output(cmd)

def mkldnn_build_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "third_party", "mkldnn", "build")


def mkl_install_dir(root):
    return os.path.join(mkldnn_build_dir(root), "intel")

def acl_root_dir(root):
    return os.path.join(mkldnn_build_dir(root), 'acl', 'ComputeLibrary')

def extra_acl_patch_dir(root):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "third_party", "bazel", "acl")

def config_mkldnn(root, args):
    build_dir = mkldnn_build_dir(root)
    ensure_empty_dir(build_dir, clear_hidden=False)
    mkl_dir = mkl_install_dir(root)
    acl_dir = acl_root_dir(root)
    acl_patch_dir = extra_acl_patch_dir(root)
    ensure_empty_dir(mkl_dir, clear_hidden=False)
    ensure_empty_dir(acl_dir, clear_hidden=False)
    if args.x86:
        with cwd(mkl_dir):
            # download mkl-lib/include
            download_cmd = """
              unset HTTPS_PROXY
              wget -nv https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/mkl/mkl-static-2022.0.1-intel_117.tar.bz2 -O mkl-static-2022.0.1-intel_117.tar.bz2
              tar xjvf mkl-static-2022.0.1-intel_117.tar.bz2
	      rm -rf mkl-static-2022.0.1-intel_117.tar.bz2
              wget -nv https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/mkl/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2 -O mkl-include-2022.0.1-h8d4b97c_803.tar.bz2
              tar xjvf mkl-include-2022.0.1-h8d4b97c_803.tar.bz2
	      rm -rf mkl-include-2022.0.1-h8d4b97c_803.tar.bz2
            """
            execute(download_cmd)

    if args.aarch64:
        with cwd(acl_dir):
            arch =  args.target_cpu_arch or "arm64-v8a"
            # downlaod and build acl for onednn
            cmd = '''
              readonly ACL_REPO="https://github.com/ARM-software/ComputeLibrary.git"
              MAKE_NP="-j$(grep -c processor /proc/cpuinfo)"

              ACL_DIR={}
              git clone --branch v23.02.1 --depth 1 $ACL_REPO $ACL_DIR
              cd $ACL_DIR
              EXTRA_ACL_PATCH_DIR={}
              for file in $EXTRA_ACL_PATCH_DIR/acl_*.patch
              do
                  if [[ $file == *makefile* ]]; then
                      continue
                  fi
                  patch -p1 < $file
              done

              scons --silent $MAKE_NP Werror=0 debug=0 neon=1 opencl=0 openmp=1 embed_kernels=0 os=linux arch={} build=native extra_cxx_flags="-fPIC"

              exit $?
            '''.format(acl_dir, acl_patch_dir, arch)
            execute(cmd)
            # a workaround for static linking
            execute('rm -f build/*.so')
            execute('mv build/libarm_compute-static.a build/libarm_compute.a')
            execute('mv build/libarm_compute_core-static.a build/libarm_compute_core.a')
            execute('mv build/libarm_compute_graph-static.a build/libarm_compute_graph.a')

    with cwd(build_dir):
        cc = which("gcc")
        cxx = which("g++")
        flags = " -DMKL_ROOT={} ".format(mkl_dir)
        envs = " CC={} CXX={} ".format(cc, cxx)
        if args.aarch64:
            envs += " ACL_ROOT_DIR={} ".format(acl_dir)
            flags += " -DDNNL_AARCH64_USE_ACL=ON "

        if args.ral_cxx11_abi:
            flags += " -DUSE_CXX11_ABI=ON"

        cmake_cmd = "{}  cmake .. {}".format(envs, flags)

        logger.info("configuring mkldnn ......")
        execute(cmake_cmd)
        logger.info("mkldnn configure success.")

@time_stage()
def build_mkldnn(root):
    build_dir = mkldnn_build_dir(root)
    with cwd(build_dir):
        execute("make -j")
        execute("make install")
    logger.info("Stage [build_mkldnn] success.")

def is_x86():
    import platform
    # TODO(disc): fine-grained check for intel/amd64
    return platform.processor() == 'x86_64'

def is_aarch64():
    import platform
    return platform.processor() == 'aarch64'

def num_make_jobs():
    cpu_cnt = os.cpu_count() or 1
    return min(max(1, cpu_cnt - 1), 32)

def auto_detect_host_cpu(args):
    if not hasattr(args, 'x86') or not args.x86:
        args.x86 = is_x86()
    if not hasattr(args, 'aarch64') or not args.aarch64:
        args.aarch64 = is_aarch64()

    enabled_cpus = int(args.x86) + int(args.aarch64)
    if enabled_cpus > 1:
        raise RuntimeError("invalid config: more than oen cpu type is specified")
    if enabled_cpus < 1:
        raise RuntimeError("auto_detect_host_cpu failed")

def update_cpu_specific_setting(args):
    if not hasattr(args, 'x86'):
        args.x86 = False
    if not hasattr(args, 'aarch64'):
        args.aarch64 = False
    if not hasattr(args, 'enable_mkldnn'):
        args.enable_mkldnn = False
    if not hasattr(args, 'cpu_only'):
        args.cpu_only = False
    if not hasattr(args, 'target_cpu_arch'):
        args.target_cpu_arch = ""

    if args.cpu_only:
        auto_detect_host_cpu(args)
        args.enable_mkldnn = (args.x86 or args.aarch64)

def get_tf_info(python_executable):
    output = subprocess.check_output(
        '{} -c "import tensorflow as tf; print(tf.__version__); print(\'\\n\'.join(tf.sysconfig.get_compile_flags())); print(\'\\n\'.join(tf.sysconfig.get_link_flags()))"'.format(
            python_executable
        ),
        shell=True,
    ).decode()
    lines = output.split("\n")
    major, minor, _ = lines[0].split(".")  # lines[0] is version like 1.15.0
    is_pai = "PAI" in lines[0]
    header_dir, lib_dir, lib_name, cxx11_abi = '', '', '', ''
    for line in lines[1:]:
        if line.startswith("-I"):
            header_dir = line[2:]
        elif line.startswith("-L"):
            lib_dir = line[2:]
        elif line.startswith("-l:"):  # in case of -l:libtensorflow_framework.so.1
            lib_name = line[3:]
        elif line.startswith("-l"):  # in case of -ltensorflow_framework
            lib_name = 'lib' + line[2:] + '.so'
        elif '_GLIBCXX_USE_CXX11_ABI' in line:
            cxx11_abi = line.split('=')[-1]
    PB_HEADER_FILE = "google/protobuf/stubs/common.h"
    proto_file_path = os.path.join(header_dir, PB_HEADER_FILE)
    if os.path.exists(proto_file_path):
        with open(proto_file_path, 'r') as f:
            content = f.read()
        try:
            match = re.findall("#define GOOGLE_PROTOBUF_VERSION [0-9]+", content)[0]
            raw_version = int(re.findall("[^0-9]+([0-9]+)$", match)[0])
            major_version = int(raw_version / 1000000)
            minor_version = int(raw_version / 1000) - major_version * 1000
            micro_version = raw_version - major_version * 1000000 - minor_version * 1000
            tf_pb_version = f"{major_version}.{minor_version}.{micro_version}"
        except IndexError as err:
            raise Exception("Can not find tensorflow's built-in pb version!")
    else:
        raise Exception("Can not find {PB_HEADER_FILE} in tf's include dir!")
    return major, minor, is_pai, header_dir, lib_dir, lib_name, cxx11_abi, tf_pb_version

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

    cuda_home = os.environ.get("TF_CUDA_HOME", None)
    if cuda_home:
        ver = _deduce_from_version_file(cuda_home)
        if ver is not None:
            return ver, cuda_home
        else:
            raise Exception(
                f"Failed to deduce cuda version from TF_CUDA_HOME: {cuda_home}"
            )

    ver = _deduce_from_version_file("/usr/local/cuda")
    if ver is not None:
        return ver, "/usr/local/cuda"

    all_cuda = [
        os.path.join("/usr/local", d)
        for d in os.listdir("/usr/local")
        if d.startswith("cuda-")
    ]
    if (len(all_cuda) != 1):
        logger.info("Mutiple cuda installed.")
    else:
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

def internal_root_dir():
    return os.path.join(get_source_root_dir(), os.pardir)

def tao_bridge_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "tao_bridge")

def tao_ral_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "tao_bridge", "ral")

def internal_tao_bridge_dir():
    return os.path.join(internal_root_dir(), "platform_alibaba", "tao_bridge")

def link_dirs(dst_dir, src_dir):
    execute("rm -rf {0} && ln -s {1} {0}".format(dst_dir, src_dir))

def try_import_from_platform_alibaba(module_name, disc_root=None):
    if disc_root is None:
        root = internal_root_dir()
    else:
        root = os.path.join(disc_root, os.path.pardir)
    build_scripts_dir = os.path.join(root, "platform_alibaba", "scripts", "python")
    sys.path.insert(0, build_scripts_dir)
    m = None
    try:
        import importlib
        m = importlib.import_module(module_name)
    except Exception as e:
        pass
    finally:
        sys.path.pop(0)
    return m


def add_arguments_platform_alibaba(parser):
    m = try_import_from_platform_alibaba("common_setup_internal")
    if not m: return
    m.add_arguments(parser)


def symlink_disc_files_platform_alibaba(args):
    m = try_import_from_platform_alibaba("common_setup_internal")
    if not m: return
    m.symlink_disc_files(args)


def configure_bridge_platform_alibaba(root, args):
    m = try_import_from_platform_alibaba("common_setup_internal")
    if not m: return
    m.configure_bridge(root, args)


def configure_compiler_platform_alibaba(root, args):
    m = try_import_from_platform_alibaba("common_setup_internal")
    if not m: return
    m.configure_compiler(root, args)

def build_tao_compiler_add_flags_platform_alibaba(root, args, flag):
    m = try_import_from_platform_alibaba("common_setup_internal", root)
    if not m: return flag
    return m.build_tao_compiler_add_flags(root, args, flag)


def test_tao_compiler_add_flags_platform_alibaba(root, args, flag):
    m = try_import_from_platform_alibaba("common_setup_internal", root)
    if not m: return flag
    return m.test_tao_compiler_add_flags(root, args, flag)


def symlink_disc_files(args):
    dir_tf_community = os.path.join(get_source_root_dir(), "tf_community")
    dir_platform_alibaba = os.path.join(internal_root_dir(), "platform_alibaba")

    logger.info("linking via tao_compiler/file_map ...")
    # map compiler codes into tf tree for build
    mapping_file = os.path.join(get_source_root_dir(), "tao_compiler", "file_map")
    with open(mapping_file) as fh:
        for line in fh:
            if line.startswith("#") or line.strip() == "":
                continue
            info = line.strip().split(",")
            if len(info) != 2:
                continue
            src_file = os.path.join(get_source_root_dir(), "tao_compiler", info[0])
            link_in_tf = os.path.join(get_source_root_dir(), "tf_community", info[1])
            dst_folder = os.path.dirname(link_in_tf)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            execute("rm -rf {0} && ln -s {1} {0}".format(link_in_tf, src_file))

    logger.info("linking ./tao to tao_compiler/tao")
    execute(
        "rm -rf {0} && ln -s {1} {0}".format(
            os.path.join(get_source_root_dir(), "tao_compiler", "tao"),
            os.path.join(get_source_root_dir(), "tao")
        )
    )

    logger.info("linking blade_gemm")
    link_dirs(os.path.join(get_source_root_dir(), 'tao', 'blade_gemm'),
            os.path.join(dir_platform_alibaba, 'blade_gemm'))
    logger.info("linking blade_service_common")
    link_dirs(os.path.join(get_source_root_dir(), 'tao', 'third_party', 'blade_service_common'),
            os.path.join(dir_platform_alibaba, 'third_party', 'blade_service_common'))

    logger.info("cleanup tao_compiler with XLA always...")

    # def link_internal_tao_bridge(is_platform_alibaba):
    # softlink ["tao_launch_op", "gpu"] dirs, and "transform" dirs are not needed for now.
    for dir_name in ["tao_launch_op", "gpu"]:
        src_file = os.path.join(internal_tao_bridge_dir(), dir_name)
        link_in_bridge = os.path.join(tao_bridge_dir(), dir_name)
        if args.platform_alibaba:
            link_dirs(link_in_bridge, src_file)
        else:
            execute("rm -rf {0}".format(link_in_bridge))
    if args.platform_alibaba:
        logger.info("linking blade_service_common bazel files")
        src_dir = os.path.join(internal_root_dir(), "platform_alibaba", "bazel", "blade_service_common")
        dst_dir = os.path.join(get_source_root_dir(), "third_party", "bazel", "blade_service_common")
        files = os.listdir(src_dir)
        for f in files:
            link_dirs(os.path.join(dst_dir, f), os.path.join(src_dir, f))

    symlink_disc_files_platform_alibaba(args)


def symlink_disc_files_deprecated(is_platform_alibaba):
    args = argparse.Namespace()
    setattr(args, "platform_alibaba", is_platform_alibaba)
    symlink_disc_files(args)

def add_ral_link_if_not_exist():
    root = get_source_root_dir()
    RAL_DIR_IN_TF = "tao_compiler/mlir/xla"
    RAL_DIR_IN_BRIDGE = os.path.join(tao_ral_dir(), "tensorflow/compiler/mlir/xla")
    if os.path.exists(RAL_DIR_IN_BRIDGE):
        shutil.rmtree(RAL_DIR_IN_BRIDGE)
    os.makedirs(RAL_DIR_IN_BRIDGE)
    with cwd(RAL_DIR_IN_BRIDGE):
        execute("ln -s {0}/{1}/ral ral".format(root, RAL_DIR_IN_TF))


def get_version_file():
    root = get_source_root_dir()
    return os.path.join(root, "VERSION")

def add_arguments_common(parser):
    parser.add_argument(
        "--cpu_only",
        required=False,
        action="store_true",
        help="Build tao with cpu support only",
    )
    parser.add_argument(
        "--aarch64",
        required=False,
        action="store_true",
        help="Build tao with aarch64 support only",
    )
    parser.add_argument(
        "--dcu",
        required=False,
        action="store_true",
        help="Build tao with dcu support only",
    )
    parser.add_argument(
        "--rocm",
        required=False,
        action="store_true",
        help="Build tao with rocm support only",
    )
    parser.add_argument(
        "--rocm_path",
        required=False,
        default=None,
        help="Build tao where rocm locates",
    )
    parser.add_argument(
        "--platform_alibaba", action="store_true", help="build with is_platform_alibaba=True"
    )
    parser.add_argument(
        "--target_cpu_arch",
        required=False,
        default="",
        help="Specify the target architecture.",
    )
    add_arguments_platform_alibaba(parser)

def disc_conf_file(root):
    return os.path.join(root, "scripts", "ci", ".disc_conf")

def save_args_to_cache(root, args):
    """Save confs to `disc_conf_file()`"""
    with open(disc_conf_file(root), "wb") as f:
        pickle.dump(args, f)

def restore_args_from_cache(root, args):
    """Restore confs from `disc_conf_file()`"""
    with open(disc_conf_file(root), "rb") as f:
        args_tmp = pickle.load(f)
        for k, v in args_tmp.__dict__.items():
            setattr(args, k, v)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cxx11_abi",
        required=False,
        action="store_true",
        help="Build with cxx11 abi or not",
    )
    add_arguments_common(parser)
    args = parser.parse_args()
    # backward compatibility
    args.ral_cxx11_abi = args.cxx11_abi
    update_cpu_specific_setting(args)
    return args


def build_tao_compiler_add_flags_platform_alibaba_cached(root, flag):
    args = argparse.Namespace()
    restore_args_from_cache(root, args)
    return build_tao_compiler_add_flags_platform_alibaba(root, args, flag)


def test_tao_compiler_add_flags_platform_alibaba_cached(root, flag):
    args = argparse.Namespace()
    restore_args_from_cache(root, args)
    return test_tao_compiler_add_flags_platform_alibaba(root, args, flag)

def cleanup_building_env(root):
    with cwd(root):
        execute("rm -rf tf_community/.tf_configure.bazelrc")

if __name__ == "__main__":
    args = parse_args()

    root = get_source_root_dir()
    cleanup_building_env(root)
    symlink_disc_files(args)

    if args.enable_mkldnn:
        config_mkldnn(root, args)
        build_mkldnn(root)

    save_args_to_cache(root, args)

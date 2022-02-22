import argparse
import os
import time
import subprocess
import shutil
import logging
import sys
from contextlib import contextmanager
from datetime import datetime


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


def symlink_files(root):
    with cwd(root):
        logger.info("configuring tao_compiler ......")
        # map compiler codes into tf tree for build
        with open("tao_compiler/file_map") as fh:
            for line in fh:
                if line.startswith("#") or line.strip() == "":
                    continue
                info = line.strip().split(",")
                if len(info) != 2:
                    continue
                src_file = os.path.join(root, "tao_compiler", info[0])
                link_in_tf = os.path.join("tf_community", info[1])
                dst_folder = os.path.dirname(link_in_tf)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                execute("rm -rf {0} && ln -s {1} {0}".format(link_in_tf, src_file))
        logger.info("linking ./tao to tf_community/tao")
        execute(
            "rm -rf {0} && ln -s {1} {0}".format(
                os.path.join("tf_community", "tao"), os.path.join(root, "tao")
            )
        )


def mkldnn_build_dir(root=None):
    if root is None:
        root = get_source_root_dir()
    return os.path.join(root, "tao", "third_party", "mkldnn", "build")


def mkl_install_dir(root):
    return os.path.join(mkldnn_build_dir(root), "intel")


def config_mkldnn(root, cxx11_abi):
    build_dir = mkldnn_build_dir(root)
    ensure_empty_dir(build_dir, clear_hidden=False)
    mkl_dir = mkl_install_dir(root)
    ensure_empty_dir(mkl_dir, clear_hidden=False)
    # download mkl-lib/include
    with cwd(mkl_dir):
        download_cmd = """
          unset HTTPS_PROXY
          curl -fsSL https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-static-2022.0.1-intel_117.tar.bz2  | tar xjv
          curl -fsSL https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2 | tar xjv
        """
        execute(download_cmd)

    with cwd(build_dir):
        cc = which("gcc")
        cxx = which("g++")
        # always link patine statically
        cmake_cmd = "CC={} CXX={} cmake .. -DMKL_ROOT={}".format(cc, cxx, mkl_dir)
        if cxx11_abi:
            cmake_cmd += " -DUSE_CXX11_ABI=ON"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cxx11_abi",
        required=False,
        action="store_true",
        help="Build with cxx11 abi or not",
    )
    args = parser.parse_args()

    root = get_source_root_dir()
    symlink_files(root)
    config_mkldnn(root, args.cxx11_abi)
    build_mkldnn(root)

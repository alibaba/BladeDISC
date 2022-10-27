import argparse, os, shutil
from re import L
from contextlib import contextmanager
import logging as logger
import subprocess

LD = "ld.lld"
TVMLIB = "libtvm.so"

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

def execute(cmd):
    """Execute a shell command, exception raised on failure."""
    shell_setting = "set -e; set -o pipefail; "
    logger.info("Execute shell command: `" + cmd + "`, cwd: " + os.getcwd())
    subprocess.check_call(shell_setting + cmd, shell=True, executable="/bin/bash")

def root_path():
    return os.path.dirname(os.path.abspath(__file__))

def tvm_path():
    return os.path.join(root_path(), "third_party", "tvm")

def tvm_build_path():
    return os.path.join(tvm_path(), "build")

def src_tvm_lib():
    return os.path.join(tvm_build_path(), TVMLIB)

def dst_tvm_lib():
    return os.path.join(dst_tvm_pkg(), TVMLIB)

def src_tvm_pkg():
    return os.path.join(tvm_path(), "python", "tvm")

def dst_pkg():
    return os.path.join(root_path(), "disc_opt")

def dst_tvm_pkg():
    return os.path.join(dst_pkg(), "tvm")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm", default="/opt/rocm", help="Path for ROCM Library")
    parser.add_argument("--llvm", default=None, help="LLVM config path for TVM building")
    parser.add_argument("--cc", default=None, help="CC config for TVM building")
    parser.add_argument("--cxx", default=None, help="CXX config path for TVM building")
    parser.add_argument("--tao_compiler", default="tao_compiler_main", help="Specify the path of tao compiler")
    parser.add_argument("--tao_bridge", default="libtao_ops.so", help="Specify the path of tao bridge")
    args = parser.parse_args()
    if args.llvm is None:
        args.llvm = os.path.join(args.rocm, "llvm/bin/llvm-config")
    return args

def configure(args):
    ensure_empty_dir(tvm_build_path())
    with cwd(tvm_build_path()):
        cmd = ""
        if args.cc is not None:
            cmd += "CC={} ".format(args.cc)
        if args.cxx is not None:
            cmd += "CXX={} ".format(args.cxx)
        cmd += "cmake .. "
        cmd += "-DUSE_ROCM={} ".format(args.rocm)
        cmd += "-DUSE_LLVM={} ".format(args.llvm)
        execute(cmd)
    execute("rm -rf {0} && ln -s {1} {0}".format(dst_tvm_pkg(), src_tvm_pkg()))

def build():
    with cwd(tvm_build_path()):
        execute("make -j")

def package(args):
    src_lld = os.path.join(os.path.dirname(args.llvm), LD)
    dst_lld = os.path.join(dst_pkg(), LD)
    execute("cp {} {}/".format(src_tvm_lib(), dst_tvm_pkg()))
    execute("rm -rf {0} && ln -s {1} {0}".format(dst_lld, src_lld))
    if args.tao_compiler:
        shutil.copy(args.tao_compiler, dst_pkg())
    if args.tao_bridge:
        shutil.copy(args.tao_bridge, dst_pkg())
    with cwd(root_path()):
        for tmpdir in ["build", "dist", "disc_opt.egg-info"]:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        cmd = "python3 setup.py bdist_wheel --universal"
        execute(cmd)

if __name__ == "__main__":
    args = parse_args()
    configure(args)
    build()
    package(args)
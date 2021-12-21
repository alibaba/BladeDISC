import subprocess
import logging
import sys
import os
import shutil
import re
from contextlib import contextmanager

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


def __create_logger():
    """Create a logger with color."""
    # The background is set with 40 plus the number of the color, and the foreground with 30
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        'WARNING': YELLOW,
        'INFO': GREEN,
        'DEBUG': BLUE,
        'CRITICAL': YELLOW,
        'ERROR': RED
    }

    class ColoredFormatter(logging.Formatter):
        def __init__(self, msg, use_color=False):
            logging.Formatter.__init__(self, msg)
            self.use_color = use_color

        def format(self, record):
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = COLOR_SEQ % (
                    30 + COLORS[levelname]) + levelname + RESET_SEQ
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    class ColoredLogger(logging.Logger):
        FORMAT = "{}%(asctime)s{} %(levelname)19s %(message)s".format(
            BOLD_SEQ, RESET_SEQ)

        def __init__(self, name):
            logging.Logger.__init__(self, name, logging.DEBUG)
            color_formatter = ColoredFormatter(self.FORMAT,
                                               use_color=sys.stdout.isatty() and sys.stderr.isatty())
            console = logging.StreamHandler()
            console.setFormatter(color_formatter)
            self.addHandler(console)
            return

    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger('tao_ci')
    logger.setLevel(logging.INFO)
    return logger


logger = __create_logger()


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
    logger.info("clear dir: {}, clear hidden files: {}".format(
        dir, clear_hidden))
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
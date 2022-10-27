import os


TUNE_LOG = "kernel_tune.json"
TEMPTUNE_LOG = "kernel_tune.log"
KERNEL_INFO = "kernel_info.txt"
PROFILE_INFO = "kernel_profiling.json"

def get_dir(pardir, dir, env, subdir=None):
    if pardir:
        path = dir
    else:
        path = os.environ.get(env, ".")
    if subdir is not None:
        path = os.path.join(path, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise Exception("The specified path {} is not directoty.".format(path))
    return path
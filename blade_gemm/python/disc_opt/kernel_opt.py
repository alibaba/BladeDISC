import os, time
import subprocess
import sys
from .disc_logging import logger
from .common import *
import json

CODEGEN_ENTRY = "disc_kernel_gen"
  
def get_path():
    return os.path.join(os.path.dirname(sys.executable), CODEGEN_ENTRY)

def get_cache(cache):
    return get_dir(cache, os.path.join(cache, "profile_cache"), "BLADNN_COLLECT_CACHE")

def parse_profiling(cache, limit):    
    file = os.path.join(get_cache(cache), PROFILE_INFO)
    if not os.path.exists(file):
        return
    with open(file) as f:
        pro = json.load(f)
    roc_dic = {}
    for key, v in pro.items():
        if not v:
            continue
        roc_dic[key] = min([float(x) for _, x in v.items()])
    sorted_dic = sorted(roc_dic.items(), key=lambda x:x[1], reverse=True)
    # logger.info(sorted_dic)
    with open(os.path.join(get_cache(cache), KERNEL_INFO), "w") as f:
        f.write("\n".join([i[0] for i in sorted_dic[0:min(limit, len(sorted_dic))]]))
    
def sort(cache=None, limit=1000):
    os.environ["DISC_KERNEL_PROFILING"] = "1"
    cmd = "{} --mode profile --codegen rocblas".format(get_path())
    if cache is not None:
        cmd += " --cache {}".format(cache)
        kernel_path = os.path.join(cache, "")
        path = os.path.dirname(os.path.abspath(__file__))
        if path != cache:
            cmd += " --ansortunelog {}".format(os.path.join(path, "profile_cache", TUNE_LOG))
            cmd += " --autotvmtunelog {}".format(os.path.join(path, "profile_cache",  TEMPTUNE_LOG))
    res = subprocess.check_output(cmd, shell=True, stderr=sys.stderr)
    parse_profiling(cache, limit)
    
def profile(tune, degree=[50, 100, 100], cache=None):
    os.environ["DISC_KERNEL_PROFILING"] = "1"
    cmd = "{} --mode profile".format(get_path())
    if tune:
        ansorstep = degree[0]
        autotvmnotrans_step = degree[1]
        autotvmtrans_step = degree[2]
    else:
        ansorstep = 0
        autotvmnotrans_step = 0
        autotvmtrans_step = 0
    cmd += " --ansortunestep {}".format(ansorstep)
    cmd += " --autotvmtunestep {},{}".format(autotvmnotrans_step, autotvmtrans_step)
    
    if cache is not None:
        cmd += " --cache {}".format(cache)
        path = os.path.dirname(os.path.abspath(__file__))
        if path != cache:
            cmd += " --ansortunelog {}".format(os.path.join(path, "profile_cache", TUNE_LOG))
            cmd += " --autotvmtunelog {}".format(os.path.join(path, "profile_cache",  TEMPTUNE_LOG))

    res = subprocess.check_output(cmd, shell=True, stderr=sys.stderr)

def generate(cache=None):
    os.environ["DISC_KERNEL_PROFILING"] = "0"
    cmd = "{} --mode generate".format(get_path())
    if cache is not None:
        cmd += " --cache {}".format(cache)

    res = subprocess.check_output(cmd, shell=True, stderr=sys.stderr)

def optimize_kernel(tune=True, cache=None, limit=None, degree=[50, 100, 100]):
    logger.info("##### Start kernel optimization. #####")
    os.environ["DISC_PROFILING_CACHE"] = get_cache(cache)
    st = time.time()
    if limit is not None:
        sort(cache, limit)
    profile(tune, degree, cache)
    generate(cache)
    en = time.time()
    logger.info("##### Finish kernel optimization. Time : {} s. #####".format(en-st))
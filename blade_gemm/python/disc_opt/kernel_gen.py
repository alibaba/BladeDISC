import os, sys
import glob, shutil
import numpy as np
from filelock import FileLock
import subprocess
TVM_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(TVM_PATH)
os.environ["PYTHONPATH"] = ":".join([TVM_PATH, os.environ.get("PYTHONPATH", "")])

import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.utils import get_const_tuple
import time
import csv
import argparse
from .disc_logging import logger
import json
import re
from tvm.contrib import rocblas
from .common import *
from tvm import autotvm

from tvm.autotvm.task.space import SplitEntity

from tvm.topi.utils import get_const_tuple

CODEGEN_TYPES=["ansor", "topi", "autotvm_notrans", "autotvm_trans"]

class GemmInfo:
    def __init__(self, shape_a, shape_b, transa, transb, dtype):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.transa = transa
        self.transb = transb
        self.dtype = dtype
        
def get_target():
    TARGET = "rocm"
    targets = subprocess.getoutput("rocm_agent_enumerator")
    targets = targets.strip().split("\n")
    target = [] 
    for t in targets:
        if t != "gfx000" and t != "gfx900":
            target.append(t)
    if len(target) == 1:
        TARGET += " -mcpu={}".format(target[0])
    return TARGET


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "generate"], default="profile", help="Two modes: 'profile' to tune and get kernel perfromace metrics; 'generate' to final best kernel implementations")
    parser.add_argument("--tune", action="store_true", help="Whether to turn on tuning")
    parser.add_argument("--codegen", default="rocblas,ansor,topi,autotvm_notrans,autotvm_trans", help='Target codegen types, multiple types separated by comma')
    parser.add_argument("--times", type=int, default=100, help='Number of iterations running kernels to get performance metrics')
    parser.add_argument("--verbose", action="store_true", help="Whether to turn on more exectution logs")
    parser.add_argument("--warm", type=int, default=5, help="Number of iterations running kernels for warmup")
    parser.add_argument("--cache", type=str, default=None, help="Location of the cache path for the generated meta files")
    parser.add_argument("--output", type=str, default=None, help="Location of the output path for the generated final kernels")
    parser.add_argument("--end", type=int, default=None, help="End ID of collected kernels to optimize")
    parser.add_argument("--start", type=int, default=None, help="Start ID of collected kernels to optimize")
    parser.add_argument("--ansortunelog", type=str, default=None, help="Location of the existing tuning records for ansor")
    parser.add_argument("--ansortunestep", type=int, default=50, help="Number of tuning steps for ansor tuning, no larger than zero means no tuning")
    parser.add_argument("--autotvmtunestep", default="50,50", help="Number of tuning steps for autotvm tuning, no larger than zero means no tuning, first for notrans template and second for trans template, separated by comma")
    parser.add_argument("--autotvmtunelog", type=str, default=None,  help="Location of the existing tuning records for autotvm")
    return parser.parse_args()

def get_cache(args):
    return get_dir(args.cache, os.path.join(args.cache, "profile_cache"), "BLADNN_COLLECT_CACHE")

def get_final_codegen_path(args):
    return get_dir(args.output, args.output, "BLADNN_KERNEL_CACHE")

def get_tmp_codegen_path(args, codegen):
    return get_dir(args.cache, os.path.join(args.cache, "tmp_cache"), "BLADNN_CODEGEN_TMP", codegen)

def get_kernels(args):
    file = os.path.join(get_cache(args), KERNEL_INFO)
    kernels = []
    if os.path.exists(file):
        with open(file) as f:
            kernels = f.read().strip().split("\n")
    return kernels

def get_ansortunelog(args):
    file = os.path.join(get_cache(args), TUNE_LOG)
    return file

def get_autotvmtunelog(args):
    file = os.path.join(get_cache(args), TEMPTUNE_LOG )
    return file


def get_profiline_json(args):
    file = os.path.join(get_cache(args), PROFILE_INFO)
    return file

def parse_profling(args):
    file = get_profiline_json(args)
    if not os.path.exists(file):
        return
    with open(file) as f:
        pro = json.load(f)
    final_path = get_final_codegen_path(args)
    """Here pro is a dictionarry describing the performance metrics, example:
       {
         "gfx90a_gemm_11776_25_1_0_0_0_float64_float64_float64_float64" : {
            "rocblas" : 1.12,
            "ansor" : 0.56
         }
       } 
    """
    for key, v in pro.items():
        if not v:
            continue
        option = min(v, key=v.get)
        if option == "rocblas":
            continue
        pattern = os.path.join(get_tmp_codegen_path(args, option), key + "*")
        for i in glob.glob(pattern):
            basename = os.path.basename(i)
            if basename.endswith("tvm_meta.json"):
                continue
            dst = os.path.join(final_path, basename)
            shutil.move(i, dst)
    logger.info("Generate final codegen into {}".format(final_path))
    for codegen in CODEGEN_TYPES:
        shutil.rmtree(get_tmp_codegen_path(args, codegen))

def parse_gemm(key):
    items = key.split("_")
    if len(items) != 12:
        return None
    m = int(items[2])
    n = int(items[3])
    k = int(items[4])
    transa = items[5] == "1"
    transb = items[6] == "1"
    transc = items[7] == "1"
    dtype = items[8]
    shape_a = (k, m) if transa else (m, k)
    shape_b = (n, k) if transb else (k, n)
    if items[8] != items[9] or \
        items[8] != items[10] or \
            items[8] != items[11]:
            return None
    if transc:
        return None
    return GemmInfo(shape_a, shape_b, transa, transb, dtype)
    
        
def gemm_schedule(op, outs, s):
    Dense = op.output(0)
    num_thread = 64
    k = Dense.op.reduce_axis[0]
    ko, kf = s[Dense].split(k, factor=num_thread)
    DenseF = s.rfactor(Dense, kf)

    if Dense.op in s.outputs:
        Out = Dense
    else:
        Out = outs[0].op.output(0)
        s[Dense].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

    tx = s[Dense].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[Dense].bind(tx, thread_x)
    s[DenseF].compute_at(s[Dense], tx)
    s[Dense].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))
    return s


@autotvm.template("sample/gemm_splitK_noTrans")
def gemm_splitK_autotvm_notrans(sizeA, sizeB, transp_a, transp_b, dtype):
    data_A = te.placeholder(tuple(sizeA), name = "data_A", dtype=dtype)
    data_B = te.placeholder(tuple(sizeB), name = "data_B", dtype=dtype)
    gemm =  topi.nn.matmul(data_A, data_B, transpose_a=transp_a, transpose_b=transp_b, out_dtype=dtype)
    s = te.create_schedule([gemm.op])
    Dense = gemm.op.output(0)
    if Dense.op in s.outputs:
        Out = Dense
    else:
        Out = gemm[0].op.output(0)
        s[Dense].compute_at(s[Out], s[Out].op.axis[1])
    
    cfg = autotvm.get_config()
    k2 = s[Dense].op.reduce_axis[0]
    cfg.define_split("tile_k_0", k2, num_outputs=2)
    ko2, kf2 = cfg["tile_k_0"].apply(s, Dense, k2)

    # ko2, kf2 = s[Dense].split(k2, factor=64)
    DenseFF = s.rfactor(Dense, ko2)
    DenseFF_p = DenseFF
    DenseFF = s.cache_write(DenseFF_p, "local")
    AA = s.cache_read(s[DenseFF].op.input_tensors[0], "shared", [DenseFF])
   
    cfg.define_split("tile_m_0", s[DenseFF_p].op.axis[1], num_outputs=2)
    cfg.define_split("tile_n_0", s[DenseFF_p].op.axis[2], num_outputs=2)

    mo, mi = cfg["tile_m_0"].apply(s, DenseFF_p, s[DenseFF_p].op.axis[1])
    no, ni = cfg["tile_n_0"].apply(s, DenseFF_p, s[DenseFF_p].op.axis[2])
    # mo, mi = s[DenseFF_p].split(s[DenseFF_p].op.axis[1], factor=1)
    # no, ni = s[DenseFF_p].split(s[DenseFF_p].op.axis[2], factor=64)
    s[DenseFF_p].reorder(mo, no, mi, ni)
    s[DenseFF_p].bind(ni, te.thread_axis("threadIdx.x")) 
    s[DenseFF_p].bind(mi, te.thread_axis("threadIdx.y")) 

    s[DenseFF_p].bind(s[DenseFF_p].op.axis[0], te.thread_axis("blockIdx.x"))
    s[DenseFF_p].bind(mo, te.thread_axis("blockIdx.y"))
    s[DenseFF_p].bind(no, te.thread_axis("blockIdx.z"))
    s[AA].compute_at(s[DenseFF_p], no)
    s[DenseFF].compute_at(s[DenseFF_p], ni)

    _, ai = cfg["tile_n_0"].apply(s, AA, s[AA].op.axis[0])
    # _, ai = s[AA].split(s[AA].op.axis[0], factor=64)
    s[AA].bind(ai, te.thread_axis("threadIdx.x"))
        
    k3 = s[Dense].op.reduce_axis[0]

    cfg.define_split("tile_k_1", k3, num_outputs=2)
    ko3, kf3 = cfg["tile_k_1"].apply(s, Dense, k3)
    # ko3, kf3 = s[Dense].split(k3, factor=64)
    DenseFFF = s.rfactor(Dense, ko3)
    s[Dense].bind(s[Dense].op.reduce_axis[0], te.thread_axis("threadIdx.y")) 
    s[DenseFFF].compute_at(s[Dense], s[Dense].op.reduce_axis[0])
    cfg.define_split("tile_n_1", s[Dense].op.axis[1], num_outputs=2)
    no, ni = cfg["tile_n_1"].apply(s, Dense, s[Dense].op.axis[1])
    mo, mi =  s[Dense].split(s[Dense].op.axis[0], factor=1)
    #ni =   s[Dense].op.axis[1]
    s[Dense].bind(ni, te.thread_axis("threadIdx.x")) 
    s[Dense].bind(mo, te.thread_axis("blockIdx.y")) 
    s[Dense].bind(no, te.thread_axis("blockIdx.x")) 
    return s, [data_A, data_B, gemm]

@autotvm.template("sample/gemm_splitK")
def gemm_splitK_autotvm(sizeA, sizeB, transp_a, transp_b, dtype):
    data_A = te.placeholder(tuple(sizeA), name = "data_A", dtype=dtype)
    data_B = te.placeholder(tuple(sizeB), name = "data_B", dtype=dtype)
    if transp_a:
        data_A_t = topi.transform.transpose(data_A)
    else:
        data_A_t = data_A
    if not transp_b:
        data_B_t = topi.transform.transpose(data_B)
    else:
        data_B_t = data_B
    gemm =  topi.nn.matmul(data_A_t, data_B_t, transpose_a=False, transpose_b=True, out_dtype=dtype)
    s = te.create_schedule([gemm.op])
    cfg = autotvm.get_config()
    for i in range(2):
        aa = gemm.op.input_tensors[i]
        in_dims = get_const_tuple(aa.shape)
        if len(s[aa].op.input_tensors) > 0:
            m, n = s[aa].op.axis
            cfg.define_split("tile_m_{}".format(i), m, policy="power2", num_outputs=2, max_factor=128, filter=lambda x: x.size[-1] >= min(min(in_dims[0], in_dims[1])/2, 32))
            # # cfg.define_split("tile_n_{}".format(i), n, policy="power2", num_outputs=2, max_factor=128, filter=lambda x: x.size[-1] >= min(in_dims[1]/2, 32))
            mo, mi = cfg["tile_m_{}".format(i)].apply(s, aa, m)
            no, ni = cfg["tile_m_{}".format(i)].apply(s, aa, n)
            # mo, mi = s[aa].split(m, factor=8)
            # no, ni = s[aa].split(n, factor=16)
            # no, ni = cfg["tile_n_{}".format(i)].apply(s, aa, n)           
            s[aa].reorder(mo, no, mi, ni)      
            s[aa].bind(mo, te.thread_axis("blockIdx.x"))
            s[aa].bind(no, te.thread_axis("blockIdx.y"))
            c = s.cache_read(s[aa].op.input_tensors[0], "shared", aa)
            s[c].compute_at(s[aa], no)
            thread_x = te.thread_axis("threadIdx.x")
            thread_y = te.thread_axis("threadIdx.y")
            s[aa].bind(ni, thread_x)

            a, _ = s[c].split(s[c].op.axis[1], factor=1)
            s[c].bind(a, thread_x)
            # cfg.define_split("tile_mi_{}".format(i), mi, num_outputs=2)
            ao, _ = s[aa].split(mi, nparts=4)
            # ao, _ = cfg["tile_mi_{}".format(i)].apply(s, aa, mi)
            s[aa].bind(ao, thread_y)
            # ao, _ = cfg["tile_mi_{}".format(i)].apply(s, c, s[c].op.axis[0])
            ao, _ = s[c].split(s[c].op.axis[0], nparts=4)
            s[c].bind(ao, thread_y)

    Dense = gemm.op.output(0)
    reduce_dim = Dense.op.reduce_axis[0]
    cfg.define_split("tile_k", reduce_dim, num_outputs=2)
    ko, kf = cfg["tile_k"].apply(s, Dense, reduce_dim)
    # ko, kf = s[Dense].split(reduce_dim, factor=1024)
    DenseF = s.rfactor(Dense, kf)
    if Dense.op in s.outputs:
        Out = Dense
    else:
        Out = gemm[0].op.output(0)
        s[Dense].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

    tx = s[Dense].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[Dense].bind(tx, thread_x)
    s[DenseF].compute_at(s[Dense], tx)
    s[Dense].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))
    return s, [data_A, data_B, gemm]

def create_schedule(lam, sche, *shapes, dtype="float32"):
    """Take numpy arrays as args, convert them to TVM tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    dev = tvm.device("rocm", 0)
    pls = []  # placeholders
    for i, shape in enumerate(shapes):
        pls.append(te.placeholder(shape, dtype=dtype, name="pl" + str(i)))
    out = lam(*pls)
    s = te.create_schedule([out.op])
    if sche is not None:
        s = sche(out.op, out, s)
    return s, pls + [out]

def resume_search(task, log_file):
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)

def generate_key(s0, s1, t0, t1, dtype):
    keys = ["dcu" , "gemm"]
    m = s0[1] if t0 else s0[0]
    n = s1[0] if t1 else s1[1]
    k = s0[0] if t0 else s0[1]
    keys += [str(m), str(n), str(k)]
    keys += ["1" if t0 else "0", "1" if t1 else "0", "0"]
    keys += [dtype]*4  
    key = "_".join(keys)
    # logging.info("Func key {}".format(key))
    return key

def rewrite_meta(fname, sch):
    with open(fname) as f:
        info = json.load(f)
    sch = str(sch)
    pattern = r'attr \[IterVar\((.+)\:.+"ThreadIndex", "(.+)"\)\] "thread_extent" = ([0-9]+)'
    ps = re.findall(pattern, sch)
    arg_map = {}
    idx = [-1, -1, -1]
    for i in ps:
        print(i)
        if i[1] == "blockIdx.x":
            idx[0] = idx[0] + 1
        elif i[1] == "blockIdx.y":
            idx[1] = idx[1] + 1
        elif i[1] == "blockIdx.z":
            idx[2] = idx[2] + 1    
        key = "{}_{}".format(i[1], max(max(idx), 0))
        if key in arg_map:
            assert i[2] == arg_map[key]
        arg_map[key] = i[2]
    for k,v in info["func_info"].items():
        tmp = []
        suffix = "_"+k[-1]
        for i in v["launch_param_tags"]:
            tmp.append(int(arg_map[i + suffix]))
        v["launch_param_args"] = tmp
    pattern = r'allocate\((.+)\: Pointer\(global.+\), (.+), \[([0-9]+)\]\), storage_scope = global'
    ps = re.findall(pattern, sch)
    info["storage"] = {}
    for i in ps:
        info["storage"][i[0]] = [i[1] , i[2]]
    newf = fname.split(".tvm_meta.json")[0] + ".meta_for_tao.json"
    logger.debug("Generate new meta to {}".format(newf))
    with open(newf, 'w') as f:
        json.dump(info, f)
    
            
def compile(args, s, farg, dtype="float32", codegen="topi", key=None):
    ss = tvm.lower(s, farg, simple_mode=True)
    m = tvm.build(s, farg, target=get_target())
    if key and codegen in CODEGEN_TYPES:
        path = get_tmp_codegen_path(args, codegen)
        hsapath = os.path.join(path, "{}.hsaco".format(key))
        metapath = os.path.join(path, "{}.tvm_meta.json".format(key))
        if args.verbose:
            schpath = os.path.join(path, "{}.schedule.txt".format(key))
            with open(schpath, "w") as f:
                print(ss, file=f)
        m.imported_modules[0].save(hsapath, fmt="hsaco")
      
        logger.debug("Generate compiled modules into {} {}".format(hsapath, metapath))
        rewrite_meta(metapath, ss)

    # m.save("outhost.ll", fmt="ll")
    # m.save("outhost.bc", fmt="bc")
    dev = tvm.rocm(0)
    pls = []  # placeholders
    vals_nd = []  # initial values
    for ar in farg[:-1]:
        vals_nd.append(tvm.nd.array(np.zeros(get_const_tuple(ar.shape), dtype=ar.dtype), dev))
        logger.debug("Input {} : {}".format(vals_nd[-1].dtype, vals_nd[-1].shape))
        # print(vals_nd[-1].dtype)
        # print(pls[-1].dtype)
    out = farg[-1]
    out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), dev)
    # m(*vals_nd, out_nd)
    return (m, vals_nd, out_nd)

def run_once(kernel_ctx, dtype="float32"):        
    m = kernel_ctx[0]
    vals_nd = kernel_ctx[1]
    out_nd = kernel_ctx[2]
    ins = []
    for j in vals_nd:
        # j = tvm.nd.array(np.random.rand(*j.shape), tvm.device("rocm", 0))    
        j = tvm.nd.array((np.ones(j.shape)*0.5).astype(dtype), tvm.device("rocm", 0))    
        ins.append(j)
    m(*(ins + [out_nd]))
    o = out_nd.numpy()

def run(args, kernel_ctx, dtype="float32"):
    for _ in range(args.warm):
        run_once(kernel_ctx, dtype)
    st = time.time() 
    for _ in range(args.times):
        run_once(kernel_ctx, dtype)
    end = time.time()
    logger.debug("Time is {} ms".format((end-st)*1000/ args.times))
    logger.debug("Out is {}".format(kernel_ctx[2]))
    logger.debug("Out is {} : {}".format(kernel_ctx[2].dtype, kernel_ctx[2].shape))

@auto_scheduler.register_workload
def gemm_compute(shape_a, shape_b, transp_a, transp_b, dtype):
    A = te.placeholder(shape_a, name="A", dtype=dtype)
    B = te.placeholder(shape_b, name="B", dtype=dtype)
    out = topi.nn.matmul(A, B, transpose_a=transp_a, transpose_b=transp_b)
    return [A, B, out]

def get_tune_ctl(args):
    steps = args.autotvmtunestep.strip().split(",")
    try:
        steps_autotvmnotrans = int(steps[0])
        steps_autotvmtrans = int(steps[1])
    except:
        raise Exception("Autotvmsteps should be two integers separated by comma.")

    return {"sample/gemm_splitK_noTrans" : steps_autotvmnotrans, 
            "sample/gemm_splitK" : steps_autotvmtrans}
     

def autotvm_gemm(args, shape_a, shape_b, transp_a, transp_b, dtype, template="sample/gemm_splitK"):
    target = get_target()
    task = autotvm.task.create(template, args=(shape_a, shape_b, transp_a, transp_b, dtype), target=target)
    logger.debug(task.config_space)
    autotvmlog = get_autotvmtunelog(args)
    autotvmsteps = get_tune_ctl(args)
    step = autotvmsteps[template]
    if  step > 0:
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
        )
        tuner = autotvm.tuner.XGBTuner(task)
        logger.debug("Tune temp step {}".format(step))
        tuner.tune(
            n_trial=step,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(autotvmlog)],
        )
    dispatch_context = autotvm.apply_history_best(autotvmlog)
    best_config = dispatch_context.query(task.target, task.workload)
    logger.debug("Best config: {}".format(best_config))
    if "{}".format(best_config) == ",None":
        raise Exception("No temp config")

    with autotvm.apply_history_best(autotvmlog):
        with tvm.target.Target(target):
            if template == "sample/gemm_splitK":
                s, arg_bufs = gemm_splitK_autotvm(shape_a, shape_b, transp_a, transp_b, dtype)
            elif template == "sample/gemm_splitK_noTrans":
                s, arg_bufs = gemm_splitK_autotvm_notrans(shape_a, shape_b, transp_a, transp_b, dtype)
    return s, arg_bufs
    

def ansor_gemm(args, shape_a, shape_b, transp_a, transp_b, dtype):
    target = get_target()
    tune_log = get_ansortunelog(args)
    logger.debug("Tuning log save to {}.".format(tune_log))
    task = tvm.auto_scheduler.SearchTask(func=gemm_compute, args=(shape_a, shape_b, transp_a, transp_b, dtype), target=target)
    step = args.ansortunestep
    if  step > 0:
        vlevel = 0
        if args.verbose:
            vlevel = 1
        logger.info("Generate better kernel implementaion. This may take a long time. Please wait.")
        logger.debug("Start tuning for {} X {}".format(shape_a, shape_b))
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=step,
            measure_callbacks=[auto_scheduler.RecordToFile(tune_log)],
            verbose=vlevel,
        )
        task.tune(tune_option)
        logger.debug("Finish tuning for {} X {}".format(shape_a, shape_b))
    # Apply the best schedule
    sch, exec_args = task.apply_best(tune_log)
    return sch, exec_args


def generate_gemm(args, key, gemm_info, codegen):
    sa = gemm_info.shape_a
    sb = gemm_info.shape_b
    transp_a = gemm_info.transa
    transp_b = gemm_info.transb
    dtype = gemm_info.dtype        
    if codegen == "rocblas":
        s, exec_args = create_schedule(
            lambda A, B: rocblas.matmul(A, B, transa=transp_a, transb=transp_b),
            None,
            sa,
            sb,
            dtype=dtype
    )
    elif codegen == "ansor":
        s, exec_args = ansor_gemm(args, sa, sb, transp_a, transp_b, dtype)
    elif codegen == "autotvm_notrans":
        s, exec_args = autotvm_gemm(args, sa, sb, transp_a, transp_b, dtype, template="sample/gemm_splitK_noTrans") 
    elif codegen == "autotvm_trans":
        s, exec_args = autotvm_gemm(args, sa, sb, transp_a, transp_b, dtype, template="sample/gemm_splitK")     
    else:
        s, exec_args = create_schedule(
        lambda A, B: topi.nn.matmul(A, B, transpose_a=transp_a, transpose_b=transp_b),
        # lambda A, B: topi.nn.matmul(topi.transform.transpose(A), topi.transform.transpose(B), transpose_a=False, transpose_b=True),
        gemm_schedule,
        sa,
        sb,
        dtype=dtype
    )
    return compile(args, s, exec_args, dtype=dtype, key=key, codegen=codegen)

class TunelogContext():
    def __init__(self, orig_ansorlogfile, ansorlogfile, orig_autotvmlogfile, autotvmlogfile):
        self.logfile = {"ansor": ansorlogfile, "autotvm": autotvmlogfile}
        self.orig_logfile = {"ansor": orig_ansorlogfile, "autotvm": orig_autotvmlogfile}
        self.records_cnt_before = {}
    
    def pre_handle(self, codegen):
        if self.orig_logfile[codegen] is not None:
            if os.path.exists(self.logfile[codegen]):
                logger.warning("Optimization records already exists for {}!".format(codegen))
            shutil.copyfile(self.orig_logfile[codegen], self.logfile[codegen])
            with open(self.logfile[codegen]) as f:
                self.records_cnt_before[codegen] = len(f.readlines())
            logger.debug("Get {} lines records before for {}.".format(self.records_cnt_before[codegen], codegen))
    
    def post_handle(self, codegen):
        if self.orig_logfile[codegen] is not None:
            with open(self.logfile[codegen]) as f:
                lines = f.readlines()
                records_cnt_after = len(lines)
            logger.debug("Get {} lines records after for {}.".format(records_cnt_after, codegen))
            if records_cnt_after > self.records_cnt_before[codegen]:
                with FileLock(self.orig_logfile[codegen] + ".lock"):
                    with open(self.orig_logfile[codegen], "a") as f:
                        f.writelines(lines[self.records_cnt_before[codegen]:])
    
    def __enter__(self):
        self.pre_handle("ansor")
        self.pre_handle("autotvm")
    
    def __exit__(self, type, value, traceback):
        self.post_handle("ansor")
        self.post_handle("autotvm")

           
def execute():
    args = parse()
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    if args.mode == "profile":
        with TunelogContext(args.ansortunelog, get_ansortunelog(args), args.autotvmtunelog, get_autotvmtunelog(args)):
            kernels = get_kernels(args)
            if "ansor" in args.codegen or "topi" in args.codegen or "autotvm" in args.codegen:
                logger.info("##### Start optimizing total {} kernels. #####".format(len(kernels)))
            else:
                logger.info("##### Start preprocessing total {} kernels. This may take some time. Please Wait. #####".format(len(kernels)))
            for idx, i in enumerate(kernels):
                st = time.time()
                if args.start is not None:
                    if idx < args.start:
                        continue
                if args.end is not None:
                    if idx > args.end:
                        continue
                key = i.strip()
                os.environ["DISC_CURRENT_FUNC_KEY"] = key
                kernel = parse_gemm(key)
                if kernel is None:
                    continue
                codegen_types = args.codegen.strip().split(",")
                for codegen in codegen_types:
                    logger.debug("Optimize kernel try {}".format(codegen))
                    try:
                        os.environ["DISC_CURRENT_CODEGEN_TYPE"] = codegen
                        k_ctx = generate_gemm(args, key, kernel, codegen=codegen)
                        run(args, k_ctx, kernel.dtype)
                    except Exception as e:
                        logger.error("Exception in {} : {}".format(key, e))
                ed = time.time()
                if "ansor" in codegen_types or "topi" in codegen_types or "autotvm_trans" in codegen_types or "autotvm_notrans" in codegen_types:
                    logger.info("Optimize kernel {} out of {} with {:.3f} s: {}.".format(idx, len(kernels), (ed-st) , key))
                # logger.info("Finish optimize kernel {} out of {} : {}.".format(idx, len(kernels), key))

    elif args.mode == "generate":
        parse_profling(args)


if __name__ == "__main__":
    execute()
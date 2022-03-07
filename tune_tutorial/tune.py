import os, sys, subprocess
import sys, os
TVM_PATH = "/global/home/aliliang/aicompiler/bladedisc/workspace/venv/lib/python3.6/site-packages/disc_dcu"
TVM_PATH = "/home/fl237079/workspace/incubator-tvm/python"
#sys.path.append(TVM_PATH)
#os.environ["PYTHONPATH"] = ":".join([TVM_PATH, os.environ.get("PYTHONPATH", "")])


CODEGEN_ENTRY = "disc_kernel_gen"
  
cache = "cache"  
#os.environ["HIP_VISIBLE_DEVICES"] = "3"
os.environ["DISC_PROFILING_CACHE"] = "outcache"

def get_path():
    return "disc_kernel_gen"
#    return "python gen.py"
#    return os.path.join(os.path.dirname(sys.executable), CODEGEN_ENTRY)

def profile(metric=False, kernel=False, step = 0, codegen="autotvm"):
    os.environ["DISC_KERNEL_PROFILING"] = "1"
    cmd = "{} --mode profile --verbose ".format(get_path())
    if codegen:
        cmd += " --codegen {}".format(codegen)
    if step > 0:
        cmd += " --temptune --temptunestep {}".format(step)
    if cache is not None:
        cmd += " --cache {}".format(cache)
    if metric:
        cmd = "rocprof -i /global/home/aliliang/metrics.txt -o metric_out.csv " + cmd
    if kernel:
        cmd = "rocprof --hip-trace -o kernel_out.csv " + cmd
    print(cmd)
    os.system(cmd)

    t = None
    with open("outcache/kernel_profiling.json") as f:
        import json
        item = json.load(f)
        for _, v in item.items():
            for _, x in v.items():
                print("time : {}".format(x))
                t = x
    if kernel:
        with open("kernel_out.stats.csv") as f:
            lines = f.readlines()
            for i in lines[1:]:
                i = i.split(",")
                print("{} : {} ms".format(i[0], float(i[3])/1000000))


    if metric:
        with open("metric_out.csv") as f:
            lines = f.readlines()
            cols = lines[0].strip().split(",")
            idx = 0
            vals = {}
            for i in lines[-2].strip().split(","):
                vals[cols[idx]] = i
                if idx >= 16:
                    print("{} : {}".format(cols[idx], i))
                idx += 1
            idx = 0
            for i in lines[-1].strip().split(","):
                vals[cols[idx]] = i
                if idx >= 16:
                    print("{} : {}".format(cols[idx], i))
                idx += 1
    return t



def generate():
    os.environ["DISC_KERNEL_PROFILING"] = "0"
    cmd = "{} --mode generate".format(get_path())
    if cache is not None:
        cmd += " --cache {}".format(cache)
        os.system(cmd)
#    res = subprocess.check_output(cmd, shell=True, stderr=sys.stderr)
if __name__ == "__main__":
#    t = profile(step=500)
    t = profile(codegen="autotvm")
    #profile(True)
    #profile(kernel=True)
    print("kernel time : {}".format(t))
    #generate()

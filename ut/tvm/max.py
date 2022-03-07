import sys, os
TVM_PATH = "/global/home/aliliang/aicompiler/bladedisc/workspace/venv/lib/python3.6/site-packages/disc_dcu"
sys.path.append(TVM_PATH)
os.environ["PYTHONPATH"] = ":".join([TVM_PATH, os.environ.get("PYTHONPATH", "")])
import argparse
import numpy as np
import tvm
from tvm import topi
TARGET = "rocm -mcpu=gfx908"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="1,4096")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--times", type=int, default=100)
    return parser.parse_args()


def max(d0, d1, args):
    dtype = args.dtype
    target = tvm.target.Target(TARGET)
    indata= tvm.te.placeholder((d0,d1), dtype=dtype)
    max_op = topi.max(indata, axis=-1)

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_reduce_schedule(target)(max_op)
    ss = tvm.lower(s, [indata, max_op], simple_mode=True)
    with open("max_tir.s", "w") as f:
        f.write(str(ss))
    foo = tvm.build(s, [indata, max_op], target, name="max")
    print(foo.imported_modules[0].as_text())
#    foo.imported_modules[0].save("max_tvm.ll", fmt="ll")
    dev=tvm.device("rocm", 0)
    inp = (np.ones((d0,d1), dtype=dtype)*0.5)
    inp = np.linspace(0, d0*d1, num=d0*d1, dtype=dtype).reshape(d0,d1)*0.5
    outp = inp.max(axis=-1)
    print(outp)
    data_tvm = tvm.nd.array(inp, device=dev)
    out_tvm = tvm.nd.empty(shape=outp.shape, device=dev, dtype=dtype)
    for _ in range(args.times):
        foo(data_tvm, out_tvm)
    print(out_tvm)





def control():
    args = parse()
    size = [int(i) for i in args.size.strip().split(",")]
    max(size[0], size[1], args)

if __name__ == "__main__":
    control()




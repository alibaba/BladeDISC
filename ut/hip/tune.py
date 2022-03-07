import subprocess
import re, os

def run(d0, d1, warp, block):
    name = "{}_{}_{}_{}".format(d0, d1, warp, block)
    if not os.path.exists("{}.stats.csv".format(name)): 
        cmd = "hipcc -o exec max.cu -I/opt/rocm/include -O3 -DD0={} -DD1={} -Dwarp_size={} -Dlaunch_dim={}".format(d0, d1, warp, block)
        print(cmd)
        o = subprocess.check_output(cmd, shell=True)
        print(o)
        cmd = "rocprof --hip-trace -o {}.csv ./exec".format(name)
        print(cmd)
        o = subprocess.check_output(cmd, shell=True).decode("utf-8")
        print(o)
        o = re.findall(r"output: (.+)\n", o)
        print(o)
    with open("{}.stats.csv".format(name)) as f:
        lines = f.readlines()
        for l in lines:
            if "main_" in l:
                t = l.rsplit(",",4)[-2]
                print(t)
    return o, t


#2    print(o)



def exec():
    if os.path.exists("tune.log"):
        os.remove("tune.log")
    if os.path.exists("tune.res"):
        os.remove("tune.res")
    dic = {}
    for d0 in [1, 32, 64, 128, 256, 512, 1024]:
        for d1 in [512, 1024, 2048, 4096, 8192]:
            for warp in [16, 32, 64, 128]:
                for block in [64, 128, 256, 512, 1024, 2048, 4096]:
                    try:
                        res, t = run(d0, d1, warp, block)
                        name = "{}_{}_{}_{}".format(d0, d1, warp, block)
                        dic[name] = t
                        with open("tune.log", "a") as f:
                            f.write(name + " : " + t + "\n")
                        with open("tune.res", "a") as f:
                            f.write("{} : {}\n".format(name, res))
                    except:
                        pass
    print(dic)

if __name__ == "__main__":
    exec()

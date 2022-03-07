
import os, subprocess

x = [1, 8, 32, 64, 128, 256, 512, 1024, 2048]

def trans(a):
    if a.endswith("us"):
        return a[:-2]
    elif a.endswith("ms"):
        return str(float(a[:-2]) * 1000)

def exec():
    wh = {}
    wh0 = {}
    wh1 = {}
    for i in x:
        dic = {}
        dic0 = {}
        dic1 = {}
        for j in x:
            name = "softmax_{}_{}.log".format(i, j)
            cmd = "nvprof python train.py --size {},{} --times 20 --tao --type soft".format(i, j)
            print(cmd)
            if os.path.exists(name):
                pass
            else:
                f = open(name, "w")
                subprocess.call(cmd, shell=True, stderr=f)
            with open(name) as f:
                lines = f.readlines()
                for l in lines:
                    if "main_k" in l:
                        ls = l.strip().split()
                        print(ls)
                        if "GPU activities:" in l:
                            t = ls[5]
                        else:
                            t = ls[3]
                        if "multiply" in ls[-1]:
                            dic[str(j)] = t
                        elif "exponential" in ls[-1]:
                            dic0[str(j)] = t
                        elif "divide" in ls[-1]:
                            dic1[str(j)] = t
            print(dic)
            print(dic0)
            print(dic1)
        wh[str(i)] = dic
        wh0[str(i)] = dic0
        wh1[str(i)] = dic1
        print(dic)
        print(dic0)
        print(dic1)
    f=open("resnet_results_mi100.txt", "w")
    print(wh, file=f)

    with open("softmul_v100.csv", "w") as f:
        cont = ",".join([str(i) for i in x])
        cont = ","+cont + "\n"
        for i in x:
            cont += str(i)
            cont += ","
            for j in x:
                cont += trans(wh[str(i)][str(j)])
                cont += ","
            cont = cont[:-1]
            cont += "\n"

        print(cont, file =f )

    with open("softexp_v100.csv", "w") as f:
        cont = ",".join([str(i) for i in x])
        cont = ","+cont + "\n"
        for i in x:
            cont += str(i)
            cont += ","
            for j in x:
                cont += trans(wh0[str(i)][str(j)])
                cont += ","
            cont = cont[:-1]
            cont += "\n"
        print(cont, file =f )

    with open("softdiv_v100.csv", "w") as f:
        cont = ",".join([str(i) for i in x])
        cont = ","+cont + "\n"
        for i in x:
            cont += str(i)
            cont += ","
            for j in x:
                cont += trans(wh1[str(i)][str(j)])
                cont += ","
            cont = cont[:-1]
            cont += "\n"
        print(cont, file =f )







if __name__ == "__main__":
    exec()
  

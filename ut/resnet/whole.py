
import os, subprocess

x = [256, 512, 1024, 2048]
y = [1, 8, 16, 32, 128, 256]

def trans(a):
    if a.endswith("us"):
        return a[:-2]
    elif a.endswith("ms"):
        return str(float(a[:-2]) * 1000)

def exec():
    wh = {}
    for i in y:
        dic = {}
        for j in x:
            name = "resnet_b{}_c{}.log".format(i, j)
            cmd = "nvprof python train.py --type resnet --size {},{},{} --times 20 --tao".format(i,int(256*56/j),j)
            print(cmd)
            if os.path.exists(name):
                pass
            else:
                f = open(name, "w")
                subprocess.call(cmd, shell=True, stderr=f)
            with open(name) as f:
                lines = f.readlines()
                for l in lines:
                    if "main_kLoop_" in l:
                        ls = l.strip().split()
                        if "GPU activities:" in l:
                            t = ls[5]
                        else:
                            t = ls[3]
                        dic[str(j)] = t
                        print(dic)
        wh[str(i)] = dic
        print(dic)
        print(dic, file=open("whole_results_v100.txt", "w"))
    with open("resnet_v100.csv", "w") as f:
        cont = ",".join([str(i) for i in x])
        cont = ","+cont + "\n"
        for i in y:
            cont += str(i)
            cont += ","
            for j in x:
                cont += trans(wh[str(i)][str(j)])
                cont += ","
            cont = cont[:-1] 
            cont += "\n"

        print(cont, file =f )




if __name__ == "__main__":
    exec()
  

import re
import sys



def check():
    with open(sys.argv[1]) as f:
        liness = f.read()
        lines = liness.split("declare i32 @llvm.amdgcn.workgroup.id.x() #1")[0]
        lines_tail = liness.split("declare i32 @llvm.amdgcn.workgroup.id.x() #1")[1]
    alls = re.findall( r"\%([0-9]+)" ,lines)
    print(alls)
    dic = {}
    for i in alls:
        dic[i] = int(i)
    new = sorted(dic.items(), key = lambda kv: kv[1])

    idx = 0
    cnt = {}
    for k in new:
        cnt[k[0]] = idx
        idx += 1
    print(cnt)
    for k, v in cnt.items():
#        if k in ["0", "8", "1", "2", "16", "32", "256"]:
            #print("%" + k)
            lines = lines.replace("%"+k+" ", "%{} ".format(v))
            lines = lines.replace("%"+k+",", "%{},".format(v))
            lines = lines.replace("%"+k+"\n", "%{}\n".format(v))
            lines = lines.replace("%"+k+")", "%{})".format(v))
            lines = lines.replace("\n"+k+":", "{}:".format(v))
    #print(lines)
    lines = lines + "declare i32 @llvm.amdgcn.workgroup.id.x() #1" + lines_tail
    with open(sys.argv[2], "w") as f:
        f.write(lines)





if __name__ == "__main__":
    check()

        




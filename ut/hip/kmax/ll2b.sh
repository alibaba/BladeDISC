/opt/rocm/llvm/bin/llc -O0 --mtriple=amdgcn-amd-amdhsa  -amdhsa-code-object-version 4  --mcpu=gfx90a --filetype=obj -o tmp_bk.o $1
#/opt/rocm/llvm/bin/llc --mcpu=gfx90a --filetype=obj -o tmp_bk.o $1
/opt/rocm/llvm/bin/ld.lld -flavor gnu -shared tmp_bk.o -o $2

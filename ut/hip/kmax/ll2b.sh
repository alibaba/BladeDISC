/opt/rocm-4.5.0/llvm/bin/llc -O3 --mtriple=amdgcn-amd-amdhsa  -amdhsa-code-object-version 4  --mcpu=gfx908 --filetype=obj -o tmp_bk.o $1
/opt/rocm-4.5.0/llvm/bin/ld.lld -flavor gnu -shared tmp_bk.o -o $2

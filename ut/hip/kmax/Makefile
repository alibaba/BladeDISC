
compile:
	hipcc -o launch module_launch.cc
	bash mimicmax.sh

run:
	./exec

prof: compile
	rocprof --hip-trace -o debug.csv ./exec
	cat debug.stats.csv

opt:
	/opt/rocm-4.5.0/llvm/bin/opt --O3 --mtriple=amdgcn-amd-amdhsa --mcpu=gfx908 sample.bc -o sampleopt.bc 
	#/opt/rocm-4.5.0/llvm/bin/opt --O3 sample.bc -o sampleopt.bc 

llc:
	#/opt/rocm-4.5.0/llvm/bin/llc -O3 --filetype=obj -o sampleopt.o sampleopt.bc 
	#/opt/rocm-4.5.0/llvm/bin/llc -O3 --mtriple=amdgcn-amd-amdhsa  -amdhsa-code-object-version 4  --mcpu=gfx908 --filetype=obj -o sampleopt.o sampleopt.bc 
	/opt/rocm-4.5.0/llvm/bin/llc -O3 --mtriple=amdgcn-amd-amdhsa  -amdhsa-code-object-version 4  --mcpu=gfx908 --filetype=obj -o sampleopt.o sample.ll
	
lld:
	/opt/rocm-4.5.0/llvm/bin/ld.lld -flavor gnu -shared sampleopt.o -o sampleopt.hsaco
	#/opt/rocm-4.5.0/llvm/bin/ld.lld -shared sampleopt.o -o sampleopt.hsaco


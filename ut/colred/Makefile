
compile:
	hipcc -o exec kernels.cu -I/opt/rocm/include -O3 
	hipcc -o launch module_launch.cc --std=c++11
	hipcc kernel.cu -o kernel.hsaco --genco
	hipcc kernel.cu -o kernel.ll -emit-llvm -S

run:
	./exec


compile:
	hipcc -o launch module_launch.cc --std=c++11
	hipcc kernel.cu -o kernel.hsaco --genco
	hipcc kernel.cu -o kernel.ll -emit-llvm -S

run:
	./exec

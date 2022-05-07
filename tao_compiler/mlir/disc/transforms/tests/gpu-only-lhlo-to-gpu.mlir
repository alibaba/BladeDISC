// RUN: disc-opt %s \
// RUN: -disc-lhlo-legalize-roots-to-parallel-loops \
// RUN: -disc-input-inline-fusion \
// RUN: -pass-pipeline='func.func(disc-parallel-loop-tiling{parallel-loop-tile-sizes=256 with-inbound-check=true})' \
// RUN: -map-parallel-loops-to-gpu \
// RUN: -convert-parallel-loops-to-gpu \
// RUN: -disc-convert-shape-to-std \
// RUN: -lhlo-fusion-inliner \
// RUN: -gpu-launch-sink-index-computations \
// RUN: -gpu-kernel-outlining \
// RUN: -disc-revise-gpu-kernel-outlining \
// RUN: -pass-pipeline='gpu.module(convert-scf-to-cf,lower-affine,strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},disc-gpu-kernel-to-blob{gpu-sm-cc-major=3 gpu-sm-cc-minor=5 multi-cc-support=false multi-cc-support-dbg-ptx-only=false blob-annotation=gpu.binary})' -split-input-file | FileCheck %s

// CHECK-LABEL: @basic_multioutput_fusion
func @basic_multioutput_fusion(%input1: memref<?xf32, "gpu">, %input2: memref<3xi32, "cpu">, %input3: memref<?x?x?xf32, "gpu">, %tmp: memref<?x?x?xf32, "gpu">, %out_1: memref<?x?x?xf32, "gpu">, %out_2: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) {
  "lmhlo.fusion"() ({
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %tmp) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32, "gpu">, memref<3xi32, "cpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.add"(%tmp, %input3, %out_1) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.multiply"(%input3, %out_1, %out_2) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test", disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  return %out_1, %out_2 : memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
}
// CHECK:    gpu.launch_func  @basic_multioutput_fusion_kernel::@basic_multioutput_fusion_kernel
// CHECK:  gpu.module @basic_multioutput_fusion_kernel
// CHECK-SAME: attributes {gpu.binary
// CHECK:    llvm.func @basic_multioutput_fusion_kernel


// CHECK-LABEL: @need_revise_kernel_outlining
func @need_revise_kernel_outlining(%input1: memref<?x?x?xf32, "gpu">, %input2: memref<3xi32, "cpu">, %input3: memref<3xi32, "cpu">, %input4: memref<3xi32, "cpu">, %input5: memref<?x?x?xf32, "gpu">, %tmp: memref<?x?x?xf32, "gpu">, %out_1: memref<?x?x?xf32, "gpu">, %out_2: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) {
  "lmhlo.fusion"() ({
    "lmhlo.real_dynamic_slice"(%input1, %input2, %input3, %input4, %tmp) : (memref<?x?x?xf32, "gpu">, memref<3xi32, "cpu">, memref<3xi32, "cpu">, memref<3xi32, "cpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.add"(%tmp, %input5, %out_1) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.multiply"(%input5, %out_1, %out_2) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test", disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  return %out_1, %out_2 : memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
}
// CHECK:    gpu.launch_func  @need_revise_kernel_outlining_kernel::@need_revise_kernel_outlining_kernel
// CHECK:  gpu.module @need_revise_kernel_outlining_kernel
// CHECK-SAME: attributes {gpu.binary
// CHECK:    llvm.func @need_revise_kernel_outlining_kernel

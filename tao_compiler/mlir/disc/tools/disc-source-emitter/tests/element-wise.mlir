// RUN: disc-source-emitter --payload-input %s | FileCheck %s


// CHECK-LABEL: @simple_elementwise
func.func @simple_elementwise(%arg0: memref<?x?xf32, "gpu">,
                              %arg1: memref<?x?xf32, "gpu">,
                              %arg2: memref<?x?xf32, "gpu">,
                              %arg3: memref<?x?xf32, "gpu">)
      -> (memref<?x?xf32, "gpu">) {
  // CHECK: "float [[ABS:.*]] = abs([[ARG0:.*]])"
  // CHECK: "float [[ADD:.*]] = [[ABS]] + [[ABS]]"
  // CHECK: "float [[CLAMP:.*]] = [[ABS]] < [[ARG0]] ? [[ARG0]] : ([[ABS]] > [[ADD]] ? [[ADD]] : [[ABS]])"
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg1, %arg2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.clamp"(%arg0, %arg1, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg3 : memref<?x?xf32, "gpu">
}
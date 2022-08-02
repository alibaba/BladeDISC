// RUN: disc-opt -disc-to-llvm -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: test_printf
module @main attributes {gpu.container_module}  {
  func.func @test_printf(%arg0: !disc_ral.context) {
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    // CHECK: llvm.mlir.addressof @"DiscPrintfDisc_Debug %d %d %d\0A" : !llvm.ptr<array<20 x i8>>
    // CHECK: llvm.call @printf({{.*}}) : (!llvm.ptr<i8>, i64, i64, i64) -> i32
    "lmhlo_disc.printf"(%c16, %c17, %c18) {format = "Disc_Debug %d %d %d\0A"} : (index, index, index) -> ()
    return
  }
}
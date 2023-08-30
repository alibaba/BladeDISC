func.func public @add_kernel_no_mask_0d1d2d3d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  %c0_2 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %dim = memref.dim %arg0[%c0_2]
  %9 = arith.muli %c1, %c256 : index
  scf.parallel (%arg1) = (%c0) to (%dim) step (%9) {
    scf.parallel (%arg2) = (%c0_2) to (%9) step (%c1) {
      %10 = arith.addi %arg2, %arg1 : index
      %true = arith.constant true
      %11 = arith.muli %arg2, %c1 : index
      %12 = arith.addi %11, %arg1 : index
      %13 = arith.cmpi ult, %12, %3 : index
      %14 = arith.andi %true, %13 : i1
      scf.if %14 {
        %15 = arith.select %4, %10, %c0 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [%dim_0], strides: [%c1] : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
        memref.assume_alignment %reinterpret_cast_3, 16 : memref<?xf32, "gpu">
        %16 = memref.load %reinterpret_cast_3[%15] : memref<?xf32, "gpu">
        %17 = arith.select %5, %10, %c0 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [%dim], strides: [%c1] : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
        memref.assume_alignment %reinterpret_cast_4, 16 : memref<?xf32, "gpu">
        %18 = memref.load %reinterpret_cast_4[%17] : memref<?xf32, "gpu">
        %19 = arith.addf %16, %18 : f32
        %reinterpret_cast_5 = memref.reinterpret_cast %alloc to offset: [%c0], sizes: [%3], strides: [%c1] : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
        memref.store %19, %reinterpret_cast_5[%10] : memref<?xf32, "gpu">
      }
      scf.yield
    }
    scf.yield
  }
  "lmhlo.terminator"() : () -> ()
}
        

func.func public @add_kernel_no_mask_0d1d2d3d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  llvm.getelementptr %arg0[%index]
  %range = memref.alloc() : memref<256xi32>
  %alloc = memref.alloc(%3) {kDiscSymbolicDimAttr = [@S2]} : memref<?xf32, "gpu">
  scf.parallel
  gpu.launch blocks(%bx, %by, %bz) in ()
             threads(%tx, %ty, %tz) in () {
    %0 = memref.load %memref_arg0[%14] : memref<?xf32>
    %1 = memref.load %memref_arg1[%15] : memref<?xf32>
    %1 = arith.addf %0, %1 : f32
    memref.store %21, %1 : memref<?xf32> 
  }

}
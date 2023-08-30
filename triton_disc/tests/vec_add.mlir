
func.func public @add_kernel_no_mask_0d1d2d3d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  %c256_i32 = arith.constant 256 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c256_i32 : i32
  %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %3 = tt.splat %1 : (i32) -> tensor<256xi32>
  %4 = arith.addi %3, %2 : tensor<256xi32>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %7 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
  %8 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
  %9 = tt.addptr %8, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
  %11 = arith.addf %7, %10 : tensor<256xf32>
  %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
  %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  tt.store %13, %11 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32>
  return
}
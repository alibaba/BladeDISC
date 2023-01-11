#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0) -> (d0 * 768)>
#map3 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map4 = affine_map<(d0) -> (8, d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (-d0 + 768, 204)>
#map7 = affine_map<(d0, d1) -> (8, d0 - d1)>
#map8 = affine_map<()[s0] -> (s0 floordiv 8)>
#map9 = affine_map<(d0, d1) -> (12, d0 - d1)>
#map10 = affine_map<()[s0] -> (s0 floordiv 12)>
#map11 = affine_map<(d0) -> (12, d0)>
module {
  func.func @main_kTransform_dot_general__2_1_0(%arg0: memref<?x3072xf32, "cpu">, %arg1: memref<3072x768xf32, "cpu">, %arg2: memref<?x768xf32, "cpu">) -> memref<?x768xf32, "cpu"> {
    %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8x12xf32>
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %true = arith.constant true
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3072 = arith.constant 3072 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    %c204 = arith.constant 204 : index
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %dim = memref.dim %arg0, %c0 : memref<?x3072xf32, "cpu">
    %0 = disc_linalg_ext.constant_wrapper dense<-8.000000e-01> : tensor<64x3072x1x12xf32>
    %1 = bufferization.to_memref %0 : memref<64x3072x1x12xf32, "cpu">
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    scf.parallel (%arg3, %arg4) = (%c0_2, %c0_2) to (%c1, %c1) step (%c1_3, %c1_3) {
      %alloca = memref.alloca() {alignment = 32 : i64} : memref<8x1xf32, "cpu">
      %alloca_4 = memref.alloca() {alignment = 32 : i64} : memref<8x12xf32, "cpu">
      %alloca_5 = memref.alloca() {alignment = 32 : i64} : memref<8x12xf32, "cpu">
      %2 = affine.max #map()[%dim]
      %3 = affine.apply #map1(%arg3)[%dim]
      %4 = affine.apply #map2(%arg4)
      %subview = memref.subview %arg2[%3, %4] [%2, 768] [1, 1] : memref<?x768xf32, "cpu"> to memref<?x768xf32, strided<[768, 1], offset: ?>, "cpu">
      %alloca_6 = memref.alloca() {alignment = 64 : i64} : memref<8x12xf32, "cpu">
      linalg.fill ins(%cst_1 : f32) outs(%alloca_6 : memref<8x12xf32, "cpu">)
      %5 = affine.apply #map3()[%2]
      %alloc = memref.alloc(%5) {alignment = 64 : i64} : memref<?x512x1x8xf32, "cpu">
      scf.for %arg5 = %c0 to %c3072 step %c512 {
        %subview_7 = memref.subview %arg0[%3, %arg5] [%2, 512] [1, 1] : memref<?x3072xf32, "cpu"> to memref<?x512xf32, strided<[3072, 1], offset: ?>, "cpu">
        scf.for %arg6 = %c0 to %2 step %c8 {
          %7 = arith.subi %2, %arg6 : index
          %8 = affine.min #map4(%7)
          %9 = arith.divsi %arg6, %c8 : index
          %10 = arith.cmpi sge, %8, %c8 : index
          scf.for %arg7 = %c0 to %c512 step %c1 {
            %subview_8 = memref.subview %subview_7[%arg6, %arg7] [%8, 1] [1, 1] : memref<?x512xf32, strided<[3072, 1], offset: ?>, "cpu"> to memref<?x1xf32, strided<[3072, 1], offset: ?>, "cpu">
            %11 = scf.if %10 -> (memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">) {
              %cast = memref.cast %subview_8 : memref<?x1xf32, strided<[3072, 1], offset: ?>, "cpu"> to memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">
              scf.yield %cast : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">
            } else {
              linalg.fill ins(%cst_1 : f32) outs(%alloca : memref<8x1xf32, "cpu">)
              %subview_9 = memref.subview %subview_8[0, 0] [%8, 1] [1, 1] : memref<?x1xf32, strided<[3072, 1], offset: ?>, "cpu"> to memref<?x1xf32, strided<[3072, 1], offset: ?>, "cpu">
              %subview_10 = memref.subview %alloca[0, 0] [%8, 1] [1, 1] : memref<8x1xf32, "cpu"> to memref<?x1xf32, strided<[1, 1]>, "cpu">
              linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%subview_9 : memref<?x1xf32, strided<[3072, 1], offset: ?>, "cpu">) outs(%subview_10 : memref<?x1xf32, strided<[1, 1]>, "cpu">) {
              ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
              }
              %cast = memref.cast %alloca : memref<8x1xf32, "cpu"> to memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">
              scf.yield %cast : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">
            }
            %12 = vector.transfer_read %11[%c0, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %13 = vector.transfer_read %11[%c1, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %14 = vector.transfer_read %11[%c2, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %15 = vector.transfer_read %11[%c3, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %16 = vector.transfer_read %11[%c4, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %17 = vector.transfer_read %11[%c5, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %18 = vector.transfer_read %11[%c6, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %19 = vector.transfer_read %11[%c7, %c0], %cst_1 {in_bounds = [true]} : memref<?x1xf32, strided<[?, 1], offset: ?>, "cpu">, vector<1xf32>
            %20 = vector.extract %12[0] : vector<1xf32>
            %21 = vector.insert %20, %cst [0, 0] : f32 into vector<1x8xf32>
            %22 = vector.extract %13[0] : vector<1xf32>
            %23 = vector.insert %22, %21 [0, 1] : f32 into vector<1x8xf32>
            %24 = vector.extract %14[0] : vector<1xf32>
            %25 = vector.insert %24, %23 [0, 2] : f32 into vector<1x8xf32>
            %26 = vector.extract %15[0] : vector<1xf32>
            %27 = vector.insert %26, %25 [0, 3] : f32 into vector<1x8xf32>
            %28 = vector.extract %16[0] : vector<1xf32>
            %29 = vector.insert %28, %27 [0, 4] : f32 into vector<1x8xf32>
            %30 = vector.extract %17[0] : vector<1xf32>
            %31 = vector.insert %30, %29 [0, 5] : f32 into vector<1x8xf32>
            %32 = vector.extract %18[0] : vector<1xf32>
            %33 = vector.insert %32, %31 [0, 6] : f32 into vector<1x8xf32>
            %34 = vector.extract %19[0] : vector<1xf32>
            %35 = vector.insert %34, %33 [0, 7] : f32 into vector<1x8xf32>
            %36 = vector.extract %35[0] : vector<1x8xf32>
            vector.transfer_write %36, %alloc[%9, %arg7, %c0, %c0] {in_bounds = [true]} : vector<8xf32>, memref<?x512x1x8xf32, "cpu">
          }
        }
        %6 = arith.cmpi eq, %arg5, %c0 : index
        scf.for %arg6 = %c0 to %c768 step %c204 {
          %7 = affine.min #map6(%arg6)
          %subview_8 = memref.subview %subview[0, %arg6] [%2, %7] [1, 1] : memref<?x768xf32, strided<[768, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu">
          scf.for %arg7 = %c0 to %2 step %c8 {
            %8 = affine.min #map7(%2, %arg7)
            %9 = affine.apply #map8()[%arg7]
            %10 = arith.cmpi sge, %8, %c8 : index
            scf.for %arg8 = %c0 to %7 step %c12 {
              %11 = affine.min #map9(%7, %arg8)
              %12 = arith.addi %arg8, %arg6 : index
              %13 = arith.addi %12, %4 : index
              %14 = affine.apply #map10()[%13]
              %subview_9 = memref.subview %subview_8[%arg7, %arg8] [%8, %11] [1, 1] : memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu">
              %15 = scf.if %6 -> (memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">) {
                %cast = memref.cast %alloca_6 : memref<8x12xf32, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              } else {
                %cast = memref.cast %subview_9 : memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              }
              %dim_10 = memref.dim %15, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              %16 = arith.cmpi sge, %dim_10, %c8 : index
              %dim_11 = memref.dim %15, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              %17 = arith.cmpi sge, %dim_11, %c12 : index
              %18 = arith.andi %16, %17 : i1
              %19 = scf.if %18 -> (memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">) {
                scf.yield %15 : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              } else {
                linalg.fill ins(%cst_1 : f32) outs(%alloca_4 : memref<8x12xf32, "cpu">)
                %49 = affine.min #map4(%dim_10)
                %50 = affine.min #map11(%dim_11)
                %subview_12 = memref.subview %15[0, 0] [%49, %50] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                %subview_13 = memref.subview %alloca_4[0, 0] [%49, %50] [1, 1] : memref<8x12xf32, "cpu"> to memref<?x?xf32, strided<[12, 1]>, "cpu">
                linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%subview_12 : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">) outs(%subview_13 : memref<?x?xf32, strided<[12, 1]>, "cpu">) {
                ^bb0(%in: f32, %out: f32):
                  linalg.yield %in : f32
                }
                %cast = memref.cast %alloca_4 : memref<8x12xf32, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              }
              %tmp_20_0 = vector.transfer_read %19[%c0, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_20_1 = vector.transfer_read %19[%c0, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_20_2 = vector.transfer_read %19[%c0, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_22_0 = vector.transfer_read %19[%c1, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_22_1 = vector.transfer_read %19[%c1, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_22_2 = vector.transfer_read %19[%c1, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_24_0 = vector.transfer_read %19[%c2, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_24_1 = vector.transfer_read %19[%c2, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_24_2 = vector.transfer_read %19[%c2, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_26_0 = vector.transfer_read %19[%c3, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_26_1 = vector.transfer_read %19[%c3, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_26_2 = vector.transfer_read %19[%c3, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_28_0 = vector.transfer_read %19[%c4, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_28_1 = vector.transfer_read %19[%c4, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_28_2 = vector.transfer_read %19[%c4, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_30_0 = vector.transfer_read %19[%c5, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_30_1 = vector.transfer_read %19[%c5, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_30_2 = vector.transfer_read %19[%c5, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_32_0 = vector.transfer_read %19[%c6, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_32_1 = vector.transfer_read %19[%c6, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_32_2 = vector.transfer_read %19[%c6, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %tmp_34_0 = vector.transfer_read %19[%c7, %c0], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_34_1 = vector.transfer_read %19[%c7, %c4], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>
              %tmp_34_2 = vector.transfer_read %19[%c7, %c8], %cst_1 {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">, vector<4xf32>

              %36:24 = scf.for %arg9 = %c0 to %c512 step %c1 iter_args(
                    %arg10 = %tmp_20_0, %arg11 = %tmp_20_1, %arg12 = %tmp_20_2,
                    %arg13 = %tmp_22_0, %arg14 = %tmp_22_1, %arg15 = %tmp_22_2,
                    %arg16 = %tmp_24_0, %arg17 = %tmp_24_1, %arg18 = %tmp_24_2,
                    %arg19 = %tmp_26_0, %arg20 = %tmp_26_1, %arg21 = %tmp_26_2,
                    %arg22 = %tmp_28_0, %arg23 = %tmp_28_1, %arg24 = %tmp_28_2,
                    %arg25 = %tmp_30_0, %arg26 = %tmp_30_1, %arg27 = %tmp_30_2,
                    %arg28 = %tmp_32_0, %arg29 = %tmp_32_1, %arg30 = %tmp_32_2,
                    %arg31 = %tmp_34_0, %arg32 = %tmp_34_1, %arg33 = %tmp_34_2
                    ) -> (
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                        vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
                    ) {
                %49 = arith.addi %arg9, %arg5 : index

                %tmp_50_0 = vector.transfer_read %alloc[%9, %arg9, %c0, %c0], %cst_1 {in_bounds = [true]} : memref<?x512x1x8xf32, "cpu">, vector<4xf32>
                %tmp_50_1 = vector.transfer_read %alloc[%9, %arg9, %c0, %c4], %cst_1 {in_bounds = [true]} : memref<?x512x1x8xf32, "cpu">, vector<4xf32>

                %tmp_51_0 = vector.transfer_read %1[%14, %49, %c0, %c0], %cst_1 {in_bounds = [true]} : memref<64x3072x1x12xf32, "cpu">, vector<4xf32>
                %tmp_51_1 = vector.transfer_read %1[%14, %49, %c0, %c4], %cst_1 {in_bounds = [true]} : memref<64x3072x1x12xf32, "cpu">, vector<4xf32>
                %tmp_51_2 = vector.transfer_read %1[%14, %49, %c0, %c8], %cst_1 {in_bounds = [true]} : memref<64x3072x1x12xf32, "cpu">, vector<4xf32>

                %tmp_52_0 = vector.extractelement %tmp_50_0[%c0 : index] : vector<4xf32>
                %tmp_52_1 = vector.broadcast %tmp_52_0 : f32 to vector<4xf32>
                %tmp_52_2 = vector.fma %tmp_52_1, %tmp_51_0, %arg10 : vector<4xf32>
                %tmp_52_3 = vector.fma %tmp_52_1, %tmp_51_1, %arg11 : vector<4xf32>
                %tmp_52_4 = vector.fma %tmp_52_1, %tmp_51_2, %arg12 : vector<4xf32>

                %tmp_53_0 = vector.extractelement %tmp_50_0[%c1 : index] : vector<4xf32>
                %tmp_53_1 = vector.broadcast %tmp_53_0 : f32 to vector<4xf32>
                %tmp_53_2 = vector.fma %tmp_53_1, %tmp_51_0, %arg13 : vector<4xf32>
                %tmp_53_3 = vector.fma %tmp_53_1, %tmp_51_1, %arg14 : vector<4xf32>
                %tmp_53_4 = vector.fma %tmp_53_1, %tmp_51_2, %arg15 : vector<4xf32>

                %tmp_56_0 = vector.extractelement %tmp_50_0[%c2 : index] : vector<4xf32>
                %tmp_56_1 = vector.broadcast %tmp_56_0 : f32 to vector<4xf32>
                %tmp_56_2 = vector.fma %tmp_56_1, %tmp_51_0, %arg16 : vector<4xf32>
                %tmp_56_3 = vector.fma %tmp_56_1, %tmp_51_1, %arg17 : vector<4xf32>
                %tmp_56_4 = vector.fma %tmp_56_1, %tmp_51_2, %arg18 : vector<4xf32>

                %tmp_57_0 = vector.extractelement %tmp_50_0[%c3 : index] : vector<4xf32>
                %tmp_57_1 = vector.broadcast %tmp_57_0 : f32 to vector<4xf32>
                %tmp_57_2 = vector.fma %tmp_57_1, %tmp_51_0, %arg19 : vector<4xf32>
                %tmp_57_3 = vector.fma %tmp_57_1, %tmp_51_1, %arg20 : vector<4xf32>
                %tmp_57_4 = vector.fma %tmp_57_1, %tmp_51_2, %arg21 : vector<4xf32>

                %tmp_58_0 = vector.extractelement %tmp_50_1[%c0 : index] : vector<4xf32>
                %tmp_58_1 = vector.broadcast %tmp_58_0 : f32 to vector<4xf32>
                %tmp_58_2 = vector.fma %tmp_58_1, %tmp_51_0, %arg22 : vector<4xf32>
                %tmp_58_3 = vector.fma %tmp_58_1, %tmp_51_1, %arg23 : vector<4xf32>
                %tmp_58_4 = vector.fma %tmp_58_1, %tmp_51_2, %arg24 : vector<4xf32>

                %tmp_59_0 = vector.extractelement %tmp_50_1[%c1 : index] : vector<4xf32>
                %tmp_59_1 = vector.broadcast %tmp_59_0 : f32 to vector<4xf32>
                %tmp_59_2 = vector.fma %tmp_59_1, %tmp_51_0, %arg25 : vector<4xf32>
                %tmp_59_3 = vector.fma %tmp_59_1, %tmp_51_1, %arg26 : vector<4xf32>
                %tmp_59_4 = vector.fma %tmp_59_1, %tmp_51_2, %arg27 : vector<4xf32>

                %tmp_60_0 = vector.extractelement %tmp_50_1[%c2 : index] : vector<4xf32>
                %tmp_60_1 = vector.broadcast %tmp_60_0 : f32 to vector<4xf32>
                %tmp_60_2 = vector.fma %tmp_60_1, %tmp_51_0, %arg28 : vector<4xf32>
                %tmp_60_3 = vector.fma %tmp_60_1, %tmp_51_1, %arg29 : vector<4xf32>
                %tmp_60_4 = vector.fma %tmp_60_1, %tmp_51_2, %arg30 : vector<4xf32>

                %tmp_61_0 = vector.extractelement %tmp_50_1[%c3 : index] : vector<4xf32>
                %tmp_61_1 = vector.broadcast %tmp_61_0 : f32 to vector<4xf32>
                %tmp_61_2 = vector.fma %tmp_61_1, %tmp_51_0, %arg31 : vector<4xf32>
                %tmp_61_3 = vector.fma %tmp_61_1, %tmp_51_1, %arg32 : vector<4xf32>
                %tmp_61_4 = vector.fma %tmp_61_1, %tmp_51_2, %arg33 : vector<4xf32>

                scf.yield
                  %tmp_52_2, %tmp_52_3, %tmp_52_4,
                  %tmp_53_2, %tmp_53_3, %tmp_53_4,
                  %tmp_56_2, %tmp_56_3, %tmp_56_4,
                  %tmp_57_2, %tmp_57_3, %tmp_57_4,
                  %tmp_58_2, %tmp_58_3, %tmp_58_4,
                  %tmp_59_2, %tmp_59_3, %tmp_59_4,
                  %tmp_60_2, %tmp_60_3, %tmp_60_4,
                  %tmp_61_2, %tmp_61_3, %tmp_61_4
                  : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                    vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                    vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                    vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                    vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                    vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
              }
              %37 = arith.cmpi sge, %11, %c12 : index
              %38 = arith.andi %10, %37 : i1
              %39 = scf.if %38 -> (memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">) {
                %cast = memref.cast %subview_9 : memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              } else {
                %cast = memref.cast %alloca_5 : memref<8x12xf32, "cpu"> to memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
                scf.yield %cast : memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              }

              vector.transfer_write %36#0, %39[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#1, %39[%c0, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#2, %39[%c0, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#3, %39[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#4, %39[%c1, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#5, %39[%c1, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#6, %39[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#7, %39[%c2, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#8, %39[%c2, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#9, %39[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#10, %39[%c3, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#11, %39[%c3, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#12, %39[%c4, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#13, %39[%c4, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#14, %39[%c4, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#15, %39[%c5, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#16, %39[%c5, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#17, %39[%c5, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#18, %39[%c6, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#19, %39[%c6, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#20, %39[%c6, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              vector.transfer_write %36#21, %39[%c7, %c0] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#22, %39[%c7, %c4] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">
              vector.transfer_write %36#23, %39[%c7, %c8] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>, "cpu">

              %48 = arith.xori %38, %true : i1
              scf.if %48 {
                %subview_12 = memref.subview %alloca_5[0, 0] [%8, %11] [1, 1] : memref<8x12xf32, "cpu"> to memref<?x?xf32, strided<[12, 1]>, "cpu">
                %subview_13 = memref.subview %subview_9[0, 0] [%8, %11] [1, 1] : memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu"> to memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu">
                linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%subview_12 : memref<?x?xf32, strided<[12, 1]>, "cpu">) outs(%subview_13 : memref<?x?xf32, strided<[768, 1], offset: ?>, "cpu">) {
                ^bb0(%in: f32, %out: f32):
                  linalg.yield %in : f32
                }
              }
            }
          }
        }
      }
      memref.dealloc %alloc : memref<?x512x1x8xf32, "cpu">
      scf.yield
    }
    return %arg2 : memref<?x768xf32, "cpu">
  }
}


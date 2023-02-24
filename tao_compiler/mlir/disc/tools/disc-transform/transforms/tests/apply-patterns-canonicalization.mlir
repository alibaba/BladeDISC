// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @fold_extracted_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @fold_extracted_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xf32> {
  // CHECK: %[[RET:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[ARG2]], %[[ARG1]]
  // CHECK-SAME: %[[ARG3]], %[[ARG4]]
  // CHECK-SAME: 1, 1
  // CHECK-NOT: tensor.extract_slice
  // CHECK: return %[[RET]]
  %0 = tensor.extract_slice %arg0[0, %arg1][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%arg2, 0][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @full_selected_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @full_selected_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = tensor.extract_slice
  // CHECK: %[[T1:.*]] = linalg.fill
  // CHECK-SAME: outs(%[[T0]] : tensor<?x?xf32>)
  // CHECK-NOT: tensor.extract_slice
  // CHECK: return %[[T1]]
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.extract_slice %arg0[%arg1, %arg2][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.extract_slice %1[0, 0][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @self_assigned_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: vector<6x16xf32>)
func.func @self_assigned_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %vec: vector<6x16xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = tensor.extract_slice
  // CHECK: %[[T1:.*]] = linalg.fill
  // CHECK-SAME: outs(%[[T0]] : tensor<?x?xf32>)
  // CHECK: %[[T2:.*]] = vector.transfer_write %[[ARG5]], %[[T1]]
  // CHECK: return %[[T2]]
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %arg0[%arg1, %arg2][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = vector.transfer_write %vec, %1[%c0, %c0] : vector<6x16xf32>, tensor<?x?xf32>
  %3 = tensor.insert_slice %2 into %1[0, 0] [%arg3, %arg4] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @transfer_read_of_fill
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: vector<6x16xf32>)
func.func @transfer_read_of_fill(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %vec: vector<6x16xf32>) -> vector<6x16xf32> {
  // CHECK: %[[RET:.*]] = arith.constant dense<0.000000e+00> : vector<6x16xf32>
  // CHECK: return %[[RET]]
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %arg0[0, 0][%arg3, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = vector.transfer_read %1[%c0, %c0], %cst : tensor<?x?xf32>, vector<6x16xf32>
  return %2 : vector<6x16xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @transfer_write_of_fill
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: vector<6x16xf32>)
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 6)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
func.func @transfer_write_of_fill(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %vec: vector<6x16xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[T1:.*]] = vector.transfer_write %[[ARG5]], %[[T0]]
  // CHECK: return %[[T1]]
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %d0 = affine.min #map1(%arg1)[%arg2]
  %d1 = affine.min #map2(%arg3)[%arg4]
  %0 = tensor.extract_slice %arg0[0, 0][%d0, %d1][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = vector.transfer_write %vec, %1[%c0, %c0] : vector<6x16xf32>, tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @fold_constant_wrapper_and_multi_level_pack
func.func @fold_constant_wrapper_and_multi_level_pack() -> tensor<64x512x1x16xf32> {
  // CHECK: %[[T0:.*]] = disc_linalg_ext.constant_wrapper dense<-8.000000e-01> : tensor<64x512x1x16xf32>
  // CHECK-NEXT: return %[[T0]] : tensor<64x512x1x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = disc_linalg_ext.constant_wrapper dense<-8.000000e-01> : tensor<512x1024xf32>
  %1 = tensor.empty() : tensor<64x512x1x16xf32>
  %2 = disc_linalg_ext.multi_level_pack %0 with padding_value(%cst : f32) tile_levels = [1, 1] tile_sizes = [1, 16] permutation = [2, 0, 1, 3] into %1 : (tensor<512x1024xf32> tensor<64x512x1x16xf32>) -> tensor<64x512x1x16xf32>
  return %2 : tensor<64x512x1x16xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

// -----

// CHECK-LABEL: @FoldXferReadOfXferWriterWithSelectPattern
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<?x?xf32>)
func.func @FoldXferReadOfXferWriterWithSelectPattern(%pred: i1, %arg0: tensor<?x?xf32>) -> vector<8x12xf32> {
  // CHECK: %[[T0:.*]] = arith.select %[[ARG0]]
  // CHECK: return %[[T0]]
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x12xf32>
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst : tensor<?x?xf32>, vector<8x12xf32>
  %1 = arith.select %pred, %cst_0, %0 : vector<8x12xf32>
  %2 = vector.transfer_write %1, %arg0[%c0, %c0] : vector<8x12xf32>, tensor<?x?xf32>
  %3 = vector.transfer_read %2[%c0, %c0], %cst : tensor<?x?xf32>, vector<8x12xf32>
  return %3 : vector<8x12xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.apply_patterns %arg1 {canonicalization}
}

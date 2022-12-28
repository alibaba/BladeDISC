transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1

  // TODO(wyzero): This actually disable multi-thread because packing for weights is not
  // supported in muti-threading config a.t.m.
  %0:2 = transform.structured.tile_to_foreach_thread_op %matmul tile_sizes [1024, 1024]
  transform.structured.fuse_into_containing_op %fill into %0#0
  %1:4 = transform.structured.tile %0#1 [6, 16, 1] {interchange=[0, 1, 2]}
  %2 = transform.structured.pad %1#0 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0], hoist_paddings=[2, 3, 0], transpose_paddings=[[1, 0], [0, 1], [0, 1]]}

  %func = transform.structured.match ops{["func.func"]} in %arg1
  transform.structured.vectorize %func {vectorize_padding}

  transform.disc.bufferize %arg1

  // TODO(wyzero): We can not merge two `lower_vectors` into one due to some wierd
  // bugs in some tile configurations. For example, it'll not converge if the tile
  // size of the k is one.
  transform.lower_vectors {
    contraction_lowering = "outerproduct",
    multireduction_lowering = "innerparallel",
    split_transfers = "linalg-copy",
    stages = [0, 1, 2, 3, 4],
    transpose_avx2_lowering = false,
    transpose_lowering = "eltwise",
    unroll_vector_transfers = true
  }

  transform.lower_vectors {
    contraction_lowering = "outerproduct",
    multireduction_lowering = "innerparallel",
    split_transfers = "linalg-copy",
    stages = [5, 6, 7],
    transpose_avx2_lowering = false,
    transpose_lowering = "eltwise",
    unroll_vector_transfers = true
  }
}
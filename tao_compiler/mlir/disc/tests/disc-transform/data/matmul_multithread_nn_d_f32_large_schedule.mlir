transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op

  // TODO(wyzero): This actually disable multi-thread because packing for weights is not
  // supported in muti-threading config a.t.m.
  %0:2 = transform.structured.tile_to_forall_op %matmul tile_sizes [1024, 1024]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %fill into %0#0
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %1:4 = transform.structured.tile %0#1 [6, 16, 1] {interchange=[0, 1, 2]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %2 = transform.structured.pad %1#0 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0], hoist_paddings=[2, 3, 0], transpose_paddings=[[1, 0], [0, 1], [0, 1]]}
    : (!transform.any_op) -> !transform.any_op

  %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.vectorize %func {vectorize_padding}

  %arg2 = transform.disc.bufferize %arg1 : (!transform.any_op) -> !transform.any_op
  transform.disc.vector.lower_vectors %arg2 contraction_lowering = outerproduct multireduction_lowering = innerparallel split_transfers = "linalg-copy" transpose_lowering = eltwise
    : (!transform.any_op) -> (!transform.any_op)
}
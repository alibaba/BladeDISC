transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op

  %0:2 = transform.structured.tile_to_forall_op %matmul tile_sizes [6, 16]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %fill into %0#0
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  transform.disc.bufferize %arg1 : (!transform.any_op) -> !transform.any_op
}
transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation

  %0:2 = transform.structured.tile_to_forall_op %matmul tile_sizes [6, 16]
  transform.structured.fuse_into_containing_op %fill into %0#0
    : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

  transform.disc.bufferize %arg1
}
transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %0:2 = transform.structured.tile_to_foreach_thread_op %matmul num_threads [1, 1]
    %1:4 = transform.structured.fuse %0#1 {tile_sizes = [8, 4, 40], tile_interchange = [0, 1, 2]}
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.disc.apply_patterns %func0 {canonicalization}
    %2 = transform.structured.pad %1#0 {padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0], hoist_paddings=[2, 3, 0], transpose_paddings=[[0, 1], [1, 0], [0, 1]]}
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.structured.vectorize %func1 {vectorize_padding}
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.disc.apply_patterns %func2 {canonicalization}
    %arg2 = transform.disc.bufferize %arg1
    %contract = transform.structured.match ops{["vector.contract"]} in %arg2 : (!pdl.operation) -> !pdl.operation
    transform.disc.vector.lower_vector_contraction_to_bfmmla_8x4x40 %contract : (!pdl.operation) -> !pdl.operation
    %func3 = transform.structured.match ops{["func.func"]} in %arg2 : (!pdl.operation) -> !pdl.operation
    transform.disc.apply_patterns %func3 {canonicalization}
    transform.disc.vector.lower_vectors %arg2 contraction_lowering = outerproduct multireduction_lowering = innerparallel split_transfers = "linalg-copy" transpose_lowering = eltwise
}

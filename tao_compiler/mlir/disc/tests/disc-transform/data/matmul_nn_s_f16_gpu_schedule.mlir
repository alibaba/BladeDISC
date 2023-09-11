transform.sequence  failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match attributes {disc.transform.name = "dot_general"} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1:2 = split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %forall_op, %tiled_op = transform.structured.tile_to_forall_op %1#1   num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %for_op, %splitted_op = transform.disc.split_reduction_serial %tiled_op by tile_sizes = [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %promoted_dot, %lhs_alloc, %rhs_alloc = transform.disc.promote_dot_operands %for_op [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %forall_op_0, %tiled_op_1 = transform.structured.tile_to_forall_op %promoted_dot   num_threads [] tile_sizes [64, 64](mapping = [#gpu.warp<x>, #gpu.warp<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %for_op_2, %splitted_op_3 = transform.disc.split_reduction_serial %tiled_op_1 by tile_sizes = [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %tiled_linalg_op, %loops:3 = transform.structured.tile %for_op_2[16, 8, 16] {interchange = [0, 1, 2]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  transform.disc.apply_licm %arg0 : !transform.any_op
  transform.disc.apply_dce %arg0 : !transform.any_op
  transform.disc.apply_cse %arg0 : !transform.any_op
  %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %3 = transform.disc.apply_patterns %2 {canonicalization} : (!transform.any_op) -> !transform.any_op
  %4 = transform.structured.vectorize %3 {vectorize_padding} : (!transform.any_op) -> !transform.any_op
  transform.disc.apply_dce %arg0 : !transform.any_op
  transform.disc.apply_cse %arg0 : !transform.any_op
  %5 = transform.disc.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
  %6 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.erase_dealloc %6 : (!transform.any_op) -> ()
  %7 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.transfer_write_zero_to_scf %7 : (!transform.any_op) -> ()
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %8 = transform.structured.match ops{["scf.forall"]} attributes {mapping = [#gpu.block<x>, #gpu.block<y>]} in %5 : (!transform.any_op) -> !transform.any_op
  %9 = transform.disc.forall_to_gpu_ctas %8 : (!transform.any_op) -> !transform.any_op
  %10 = transform.structured.match ops{["scf.forall"]} attributes {mapping = [#gpu.warp<x>, #gpu.warp<y>]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.forall_to_gpu_warps %10 : (!transform.any_op) -> ()
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %11 = transform.structured.match ops{["linalg.generic"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.gmem_to_smem %11 : (!transform.any_op) -> ()
  %12 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.vector.vector_to_mma_conversion %12 : (!transform.any_op) -> ()
  transform.disc.apply_licm %5 : !transform.any_op
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %13 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.inline_and_convert_gpu_ids %13 : (!transform.any_op) -> ()
  transform.disc.apply_licm %5 : !transform.any_op
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match attributes {disc.transform.name = "dot_general"} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1:2 = split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %forall_op, %tiled_op = transform.structured.tile_to_forall_op %1#1 num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<x>, #gpu.block<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %padding_mn = transform.disc.padding_mn %tiled_op padding_values [0.0:f16, 0.0:f16, 0.0:f16] tile_sizes [128, 128] : (!transform.any_op) -> (!transform.any_op)
  %for_op, %splitted_op = transform.disc.split_reduction_serial %padding_mn by tile_sizes = [32] loop_type = "cta-k-loop" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %padding_k = transform.disc.padding_k %for_op padding_values [0.0:f16, 0.0:f16] tile_sizes [32] : (!transform.any_op) -> (!transform.any_op)
  transform.disc.apply_dce %arg0 : !transform.any_op
  transform.disc.apply_cse %arg0 : !transform.any_op
  %promoted_dot, %lhs_alloc, %rhs_alloc = transform.disc.promote_dot_operands %padding_k [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %forall_op_0, %tiled_op_1 = transform.structured.tile_to_forall_op %promoted_dot num_threads [] tile_sizes [64, 64](mapping = [#gpu.warp<x>, #gpu.warp<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %for_op_2, %splitted_op_3 = transform.disc.split_reduction_serial %tiled_op_1 by tile_sizes = [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %tiled_linalg_op, %loops:3 = transform.structured.tile %for_op_2[16, 8, 16] {interchange = [0, 1, 2]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  transform.disc.apply_licm %arg0 : !transform.any_op
  transform.disc.apply_dce %arg0 : !transform.any_op
  transform.disc.apply_cse %arg0 : !transform.any_op
  %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %3 = transform.disc.apply_patterns %2 {canonicalization} : (!transform.any_op) -> !transform.any_op
  %4 = transform.structured.vectorize %3 {vectorize_padding} : (!transform.any_op) -> !transform.any_op
  %func1 = transform.structured.match ops{["func.func"]} in %4 : (!transform.any_op) -> !transform.any_op
  transform.disc.swap_alloc_tensor %func1 : (!transform.any_op) -> ()
  %5 = transform.disc.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
  %6 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.erase_dealloc %6 : (!transform.any_op) -> ()
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %8 = transform.structured.match ops{["scf.forall"]} attributes {mapping = [#gpu.block<x>, #gpu.block<y>]} in %5 : (!transform.any_op) -> !transform.any_op
  %9 = transform.disc.forall_to_gpu_ctas %8 : (!transform.any_op) -> !transform.any_op
  %10 = transform.structured.match ops{["scf.forall"]} attributes {mapping = [#gpu.warp<x>, #gpu.warp<y>]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.forall_to_gpu_warps %10 : (!transform.any_op) -> ()
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %12 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.vector.vector_to_mma_conversion %12 : (!transform.any_op) -> ()
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  // 1. use register to cache the result of ldmatrix
  // 2. use register to cache the result of mma's accumulation result
  // 3. store the final result from reg to smem and to gmem
  // 4. use padding for output smem matrix to avoid bank conflict`
  %mma = transform.structured.match ops{["nvgpu.mma.sync"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.move_data_to_register %mma by block_mn_shape = [128, 128] smem_padding = 8 : (!transform.any_op) -> ()
  transform.disc.apply_licm %5 : !transform.any_op
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  // use cp.asys to load matrix A and B from gmem to smem
  %transfer_write = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.expand_transfer_rw_to_memref_copy %transfer_write : (!transform.any_op) -> ()
  // swizzle the access of input matrix,
  // including from gmem to smem by cp.async and from smem to reg by ldmatrix
  %swizzle = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.swizzle_smem %swizzle : (!transform.any_op) -> ()
  // multi buffering for software pipeline
  %multi_buffering = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.multi_buffering %multi_buffering by multi_buffering_factor = 2 : (!transform.any_op) -> ()
  // reuse smem for input and output matrix
  %pack_smem = transform.structured.match ops{["scf.parallel"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.pack_smem %pack_smem : (!transform.any_op) -> ()
  // manually lowering nvgpu's DeviceAsyncCreateGroupOp and DeviceAsyncWaitOp to NVVM's correspondingly,
  // so that DeviceAsyncToken no longer cta-k-loop's loop carried variable,
  // which is easier for further software pipeline
  %14 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.convert_nvgpu_async_cp_to_nvvm_async_cp %14 : (!transform.any_op) -> ()
  // software pipeline
  %pipeline = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.gpu_software_pipeline %pipeline by depth = 2: (!transform.any_op) -> ()
  transform.disc.apply_licm %5 : !transform.any_op
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %13 = transform.structured.match ops{["func.func"]} in %5 : (!transform.any_op) -> !transform.any_op
  transform.disc.inline_and_convert_gpu_ids %13 : (!transform.any_op) -> ()
  transform.disc.apply_licm %5 : !transform.any_op
  transform.disc.apply_dce %5 : !transform.any_op
  transform.disc.apply_cse %5 : !transform.any_op
  %canonicalization1 = transform.disc.apply_patterns %5 {canonicalization} : (!transform.any_op) -> !transform.any_op 
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %foreach_thread_op, %tiled_op = transform.structured.tile_to_foreach_thread_op %matmul num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<x>, #gpu.block<y>])
  %2 = transform.structured.fuse_into_containing_op %fill into %foreach_thread_op
  %padding_mn = transform.disc.padding_mn %tiled_op padding_values [0.0:f16, 0.0:f16, 0.0:f16] tile_sizes [128, 128] : (!pdl.operation) -> (!pdl.operation)
  %promoted_acc = transform.disc.promote_dot_operands %padding_mn [2] : (!pdl.operation) -> (!pdl.operation) 
  %for_op, %splitted_op = transform.disc.split_reduction_serial %promoted_acc by tile_sizes = [32] loop_type = "cta-k-loop"
  %padding_k = transform.disc.padding_k %for_op padding_values [0.0:f16, 0.0:f16] tile_sizes [32] : (!pdl.operation) -> (!pdl.operation)
  %promoted_dot = transform.disc.promote_dot_operands %padding_k [0, 1] : (!pdl.operation) -> (!pdl.operation)
  %foreach_thread_op_0, %tiled_op_1 = transform.structured.tile_to_foreach_thread_op %promoted_dot num_threads [] tile_sizes [64, 64](mapping = [#gpu.thread<x>, #gpu.thread<y>])  
  %for_op_2, %splitted_op_3 = transform.disc.split_reduction_serial %tiled_op_1 by tile_sizes = [32] loop_type = "warp-k-loop"
  %tiled_linalg_op, %loops:3 = transform.structured.tile %for_op_2[16, 8, 16] {interchange = [0,1,2]} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %5 = transform.structured.vectorize %arg0 {vectorize_padding}
  %func1 = transform.structured.match ops{["func.func"]} in %5 : (!pdl.operation) -> !pdl.operation
  transform.disc.swap_alloc_tensor %func1 : (!pdl.operation) -> ()
  %6 = transform.disc.bufferize {target_gpu} %5
  %8 = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.transfer_write_zero_to_scf %8 : (!pdl.operation) -> ()
  %9 = transform.structured.match ops{["scf.foreach_thread"]} attributes {mapping = [#gpu.block<x>, #gpu.block<y>]} in %6 : (!pdl.operation) -> !pdl.operation
  %10 = transform.disc.foreach_thread_to_gpu_ctas %9 : (!pdl.operation) -> !pdl.operation
  %11 = transform.structured.match ops{["scf.foreach_thread"]} attributes {mapping = [#gpu.thread<x>, #gpu.thread<y>]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.foreach_thread_to_gpu_warps %11 : (!pdl.operation) -> ()
  %13 = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.vector.vector_to_mma_conversion %13 : (!pdl.operation) -> ()
  %mma1 = transform.structured.match ops{["nvgpu.mma.sync"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.use_reg_for_accumulation %mma1 by block_mn_shape = [128, 128] smem_padding = 8 : (!pdl.operation) -> ()
  %transfer_write = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.expand_transfer_rw_to_memref_copy %transfer_write use_async_copy = true alignment = 64 threadid = 0 s_offset_x 4 s_offset_y 0 g_offset_x 4 g_offset_y 0 src_elements 3 : (!pdl.operation) -> ()
  %swizzle = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.swizzle_smem %swizzle : (!pdl.operation) -> ()
  %multi_buffering = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.multi_buffering %multi_buffering by multi_buffering_factor = 2 : (!pdl.operation) -> ()
  %pack_smem = transform.structured.match ops{["scf.parallel"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.pack_smem %pack_smem alignment = 64 : (!pdl.operation) -> ()
  %pipeline = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.gpu_software_pipeline %pipeline by depth = 2 k_dim = 1024: (!pdl.operation) -> ()
  %14 = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.inline_and_convert_gpu_ids %14 : (!pdl.operation) -> ()
  %7 = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  transform.disc.erase_dealloc %7 : (!pdl.operation) -> ()
  // %15 = transform.structured.match ops{["func.func"]} in %6 : (!pdl.operation) -> !pdl.operation
  // transform.disc.convert_nvgpu_async_cp_to_nvvm_async_cp %15 : (!pdl.operation) -> ()
  }
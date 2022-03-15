// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NONE_MAX_SUPPRESSION_GPU_KERNEL_IMPL__
#define __NONE_MAX_SUPPRESSION_GPU_KERNEL_IMPL__
/*
 * None Maximum Suppression implementation on GPU device
 */

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "tensorflow/core/framework/register_types.h"
#define EIGEN_USE_GPU  // This defination is essential
#if TF_MAJOR == 2 || (TF_MAJOR == 1 && TF_MINOR >= 14)
#include "tensorflow/core/util/gpu_kernel_helper.h"
#else
#include "tensorflow/core/util/cuda_kernel_helper.h"
#endif
#else
#include <cuda_runtime_api.h>
#endif  // GOOGLE_CUDA

#include "cub/cub.cuh"
#include "src/custom_ops/non_max_suppression/cuda_common.cu.h"

#if GOOGLE_CUDA
namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

// -------------------------------------------------------------------------------------

/** None Maximum Suppression Kernel Control options **/
static const int BLOCK_MIN = 256;
static const int BLOCK_MID = 512;
static const int BLOCK_MAX = 1024;

static const int BARRIER_LAG = 1024 * 2;
static const int BARRIER_SML = 1024;
static const int MASK_SIZE = 1024;

// -------------------------------------------------------------------------------------

__global__ void prepareTempSortIndexScore(const int size, const float* scores,
                                          float* temp_scores,
                                          int* temp_pre_sort_indexes,
                                          int* temp_pos_sort_indexes) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    /** set temp scores and indices **/
    temp_scores[i] = scores[i];
    temp_pre_sort_indexes[i] = i;
    temp_pos_sort_indexes[i] = i;
  }
}

__global__ void prepareTempSortIndexScoreAndFilterScore(
    const int size, const float score_threshold, const float* scores,
    float* temp_scores, int* temp_pre_sort_indexes, int* temp_pos_sort_indexes,
    int* score_filter_kept) {
  __shared__ int block_kept[1];
  block_kept[0] = 0;
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(i, size) {
    float score = scores[i];
    /** set temp scores and indices **/
    if (score > score_threshold) {
      temp_scores[i] = score;
      temp_pre_sort_indexes[i] = i;
      temp_pos_sort_indexes[i] = i;
      atomicAdd(block_kept, 1);
    } else {
      temp_scores[i] = FLT_MIN;
      temp_pre_sort_indexes[i] = -1;
      temp_pos_sort_indexes[i] = -1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(score_filter_kept, block_kept[0]);
  }
}

void sortScoreIndexPairGpu(OpKernelContext* ctx, cudaStream_t stream,
#if GOOGLE_CUDA
                           const GPUDevice& d,
#endif  // GOOGLE_CUDA
                           const int size, const float* scores,
                           int* sorted_indexes, int* score_filter_kept_hst,
                           const bool do_filter_score = false,
                           const float score_threshold = FLT_MAX) {
  Tensor temp_pre_sort_indexes_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT32, TensorShape({static_cast<int64>(size * sizeof(int))}),
               &temp_pre_sort_indexes_tensor));
  int* temp_pre_sort_indexes = temp_pre_sort_indexes_tensor.flat<int>().data();

  Tensor temp_sorted_scores_tensor;
  OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
          DT_FLOAT, TensorShape({static_cast<int64>(size * sizeof(float))}),
          &temp_sorted_scores_tensor));
  float* temp_sorted_scores = temp_sorted_scores_tensor.flat<float>().data();

  if (do_filter_score) {
    Tensor score_filter_kept_tensor;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
            DT_INT32, TensorShape({static_cast<int64>(size * sizeof(int))}),
            &score_filter_kept_tensor));
    int* score_filter_kept = score_filter_kept_tensor.flat<int>().data();
    MEMSET_DEV(score_filter_kept, 0,
               sizeof(int));  // set score_filter_kept 0 at init.
#if GOOGLE_CUDA
    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    prepareTempSortIndexScoreAndFilterScore<<<
        config.block_count, config.thread_per_block, 0, stream>>>
#else
    prepareTempSortIndexScoreAndFilterScore<<<GET_DEFAULT_1D_GRID_SIZE(size),
                                              BLOCK_MID, 0, stream>>>
#endif  // GOOGLE_CUDA
        (size, score_threshold, scores, temp_sorted_scores,
         temp_pre_sort_indexes, sorted_indexes, score_filter_kept);
    COPY_D2H_ASYNC(score_filter_kept_hst, score_filter_kept, sizeof(int),
                   stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    /** in case score threshold filtered out all boxes, stop NMS here **/
    if ((*score_filter_kept_hst) == 0) {
      return;
    }
  } else {
#if GOOGLE_CUDA
    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    prepareTempSortIndexScore<<<config.block_count, config.thread_per_block, 0,
                                stream>>>
#else
    prepareTempSortIndexScore<<<GET_DEFAULT_1D_GRID_SIZE(size), BLOCK_MID, 0,
                                stream>>>
#endif  // GOOGLE_CUDA
        (size, scores, temp_sorted_scores, temp_pre_sort_indexes,
         sorted_indexes);
  }

  void* cub_space = NULL;
  size_t cub_space_bytes = 0;

  cub::DeviceRadixSort::SortPairsDescending(
      cub_space, cub_space_bytes, scores /* keys in */,
      temp_sorted_scores /* keys out */, temp_pre_sort_indexes /* values in */,
      sorted_indexes /* values out */, size /* num items */, 0 /* begin bit */,
      sizeof(float) * 8 /* end bit */, stream /* stream */,
      false /* debug symbol */);

  Tensor cub_space_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT32, TensorShape({static_cast<int64>(cub_space_bytes)}),
               &cub_space_tensor));
  cub_space = (void*)cub_space_tensor.flat<int>().data();

  cub::DeviceRadixSort::SortPairsDescending(
      cub_space, cub_space_bytes, scores /* keys in */,
      temp_sorted_scores /* keys out */, temp_pre_sort_indexes /* values in */,
      sorted_indexes /* values out */, size /* num items */, 0 /* begin bit */,
      sizeof(float) * 8 /* end bit */, stream /* stream */,
      false /* debug symbol */);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------------------

__forceinline__ __device__ float bboxSize(const float4& bbox) {
  if (bbox.z < bbox.x || bbox.w < bbox.y) {
    return 0.f;
  } else {
    float width = bbox.z - bbox.x;
    float height = bbox.w - bbox.y;
    return width * height;
  }
}

/*
 * do not force this function inline,
 * may cause performance drop
 */
__device__ float iouOverlap(const float4& bbox1, const float4& bbox2) {
  /** intersection bbox **/
  float4 intersect_bbox;
  if (bbox2.x > bbox1.z || bbox2.z < bbox1.x || bbox2.y > bbox1.w ||
      bbox2.w < bbox1.y) {
    /** Return [0, 0, 0, 0] if there is no intersection **/
    intersect_bbox.x = 0.f;
    intersect_bbox.y = 0.f;
    intersect_bbox.z = 0.f;
    intersect_bbox.w = 0.f;
  } else {
    intersect_bbox.x = max(bbox1.x, bbox2.x);
    intersect_bbox.y = max(bbox1.y, bbox2.y);
    intersect_bbox.z = min(bbox1.z, bbox2.z);
    intersect_bbox.w = min(bbox1.w, bbox2.w);
  }
  /** calculate IOU overlap **/
  float intersect_width, intersect_height;
  intersect_width = intersect_bbox.z - intersect_bbox.x;
  intersect_height = intersect_bbox.w - intersect_bbox.y;
  if (intersect_width > 0.f && intersect_height > 0.f) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bboxSize(bbox1);
    float bbox2_size = bboxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.f;
  }
}

/*
 * filter out boxes with IOU > nms_threshold
 * The filter are boxes determined to be kept.
 *   (previously got through NMS kernel for the front num==MASK_SIZE bounding
 * boxes.) The bounding boxes to be filtered are the original boxes ranked after
 * front num==MASK_SIZE.
 */
__global__ void nmsFilter(const int size, const int filter_size,
                          const int* filter_indexes, const float nms_threshold,
                          const float4* bboxes, int* indexes,
                          int* filter_kept) {
  /** filter bounding boxes **/
  extern __shared__ float4 ref_bboxes[];
  __shared__ int block_filter_kept[1];
  /** initilized filter bounding boxes **/
  for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
    const int filter_bbox_idx = filter_indexes[i];
    ref_bboxes[i] = bboxes[filter_bbox_idx];
  }
  block_filter_kept[0] = 0;
  __syncthreads();

  CUDA_1D_KERNEL_LOOP(i, size) {
    const int cur_bbox_idx = indexes[i];
    bool is_kept = true;
    if (cur_bbox_idx == -1) {
      is_kept = false;
    } else {
      const float4 cur_bbox = bboxes[cur_bbox_idx];
#pragma unroll
      for (int j = 0; j < filter_size; ++j) {
        if ((is_kept) &&
            (iouOverlap(ref_bboxes[j], cur_bbox)) > nms_threshold) {
          /** mark this bbox as removed **/
          is_kept = false;
          indexes[i] = -1;
        }
      }
    }
    /** do count, if current bounding box is kept after filtering */
    if (is_kept) {
      atomicAdd(block_filter_kept, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(filter_kept, block_filter_kept[0]);
  }
}

/*
 * none maximum suppression kernel body
 * We use only sinlge block for this execution, considering:
 * - If the original size (number of bounding boxes) is too large,
       We should done IOU Filter ahead, lower the size to a acceptble level.
 * - thread-block synchronization works efficiently
 * - kernel resources like shared memory, registers, constants usage
 *     are easy to control.
 */
template <int Nelem, bool KeptCount>
__global__ void nmsGpuKernel(const int size, const int out_top_k,
                             const float nms_threshold, const float4* bboxes,
                             const int* indexes, int* out_indexes,
                             int* kept_count) {
  /** flags of kept or not, for all boxes **/
  extern __shared__ bool shared_kept_bboxes[];
  /** local thread data **/
  float4 loc_bbox[Nelem];
/** initialize float4, Bboxinfo, shared_kept_bboxes **/
#pragma unroll
  for (int e = 0; e < Nelem; e++) {
    const int cur_idx = threadIdx.x + blockDim.x * e;
    if (cur_idx < size) {
      const int cur_bbox_idx = indexes[cur_idx];
      if (cur_bbox_idx != -1) {
        loc_bbox[e] = bboxes[cur_bbox_idx];
        shared_kept_bboxes[cur_idx] = true;
      } else {
        shared_kept_bboxes[cur_idx] = false;
      }
    } else {
      shared_kept_bboxes[cur_idx] = false;
    }
  }
  /** synchronize after initialization **/
  __syncthreads();
  /** filter out overlapped boxes with lower scores **/
  int kept = 0;
  int ref_indexes_idx = 0;
  int ref_bbox_idx = indexes[ref_indexes_idx];
  /*
   * reference box index == -1, means this box has been filtered out by score
   * threshold, and all boxes ranked after this box are filtered out as well
   * (since we have done sorting previously).
   */
  while ((ref_bbox_idx != -1) && (ref_indexes_idx < size) &&
         (kept < out_top_k)) {
    kept++;
    float4 ref_bbox;
    ref_bbox = bboxes[ref_bbox_idx];
/** calculate reference bbox's IOU overlap with bboxes within current thread **/
#pragma unroll
    for (int e = 0; e < Nelem; e++) {
      const int cur_idx = threadIdx.x + blockDim.x * e;
      if ((shared_kept_bboxes[cur_idx]) && (cur_idx > ref_indexes_idx)) {
        if (iouOverlap(ref_bbox, loc_bbox[e]) > nms_threshold) {
          shared_kept_bboxes[cur_idx] = false;
        }
      }
    }
    __syncthreads();
    /** write kept bbox index out **/
    if (threadIdx.x == 0) {
      out_indexes[kept - 1] = ref_bbox_idx;
    }
    /** update reference bbox **/
    do {
      ref_indexes_idx++;
    } while ((ref_indexes_idx < size) &&
             (!shared_kept_bboxes[ref_indexes_idx]) && (kept < out_top_k));
    /*
     * check ref_indexes_idx within range first,
     * otherwise may cause error in reading shared_kept_bboxes
     */
    ref_bbox_idx = indexes[ref_indexes_idx];
  }
  /** write out kept boxes count number **/
  if (KeptCount) {
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
      *kept_count = kept;
    }
  }
}

/** Explicit Templatized NMS Kernels **/
#ifdef PRINT_DEBUG
#define KnoCount(Nelem) nmsGpuKernel<(Nelem), true>
#else
#define KnoCount(Nelem) nmsGpuKernel<(Nelem), false>
#endif  // PRINT_DEBUG
void (*activeKernelsNoCount[])(const int, const int, const float, const float4*,
                               const int*, int*, int*) = {
    KnoCount(1), KnoCount(2), KnoCount(3), KnoCount(4),
    KnoCount(5), KnoCount(6), KnoCount(7),
};

#define KdoCount(Nelem) nmsGpuKernel<(Nelem), true>
void (*activeKernelsDoCount[])(const int, const int, const float, const float4*,
                               const int*, int*, int*) = {
    KdoCount(1), KdoCount(2), KdoCount(3), KdoCount(4),
    KdoCount(5), KdoCount(6), KdoCount(7),
};
/*
 * According to the resources per thread/stream-multiprocessor,
 * we could only allow per-thread to handle 7 Nelem (bounding-box)
 * at maximum.
 */

// -------------------------------------------------------------------------------------

int nmsGpuSmallSize(cudaStream_t stream, const int size, const int out_top_k,
                    const float nms_threshold, const float4* bboxes,
                    const int* indexes, int* out_indexes, int* kept_count) {
  const int BS = BLOCK_MIN;
  const int GS = 1;
  const int n_elem = DIV_UP(size, BS);
  activeKernelsDoCount[n_elem -
                       1]<<<GS, BS, BS * n_elem * sizeof(bool), stream>>>(
      size, out_top_k, nms_threshold, bboxes, indexes, out_indexes, kept_count);
  int* kept_count_hst;
  LOC_HST(kept_count_hst, sizeof(int));
  COPY_D2H_ASYNC(kept_count_hst, kept_count, sizeof(int), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int output_value = (*kept_count_hst);
  CUDA_CHECK(cudaFreeHost(kept_count_hst));
  return output_value;
}

int nmsGpuMediumSize(cudaStream_t stream, const int size, const int out_top_k,
                     const float nms_threshold, const float4* bboxes,
                     const int* indexes, int* out_indexes, int* kept_count) {
  const int BS = BLOCK_MAX;
  const int GS = 1;
  const int n_elem = DIV_UP(size, BS);
  activeKernelsDoCount[n_elem -
                       1]<<<GS, BS, BS * n_elem * sizeof(bool), stream>>>(
      size, out_top_k, nms_threshold, bboxes, indexes, out_indexes, kept_count);
  int* kept_count_hst;
  LOC_HST(kept_count_hst, sizeof(int));
  COPY_D2H_ASYNC(kept_count_hst, kept_count, sizeof(int), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int output_value = (*kept_count_hst);
  CUDA_CHECK(cudaFreeHost(kept_count_hst));
  return output_value;
}

int nmsGpuLargeSize(OpKernelContext* ctx, cudaStream_t stream,
#if GOOGLE_CUDA
                    const GPUDevice& d,
#endif  // GOOGLE_CUDA
                    const int size, const int out_top_k,
                    const float nms_threshold, const float4* bboxes,
                    int* indexes, int* out_indexes, int* kept_count) {
  const int BS_mask = BLOCK_MIN;
  const int GS_mask = 1;
  const int n_elem = DIV_UP(MASK_SIZE, BS_mask);

  int p_size = size;
  int p_offset = 0;
  int total_kept_count = 0;

  while (p_size > BARRIER_LAG) {
    /*
     * When the number of boxes is too large (exceeds BARRIER_LAG), assuming it
     * is 'N'. we do a NMS Filter first:
     *   1. do NMS for the front 'M' (equals MASK_SIZE) bboxes, got 'K' kept
     * bboxes from the front 'M' bboxes.
     *   2. use these 'K' bboxes to filter out the left ['N' - 'M'] bboxes.
     *   3. assuming we got 'L' bboxes left from step 2.
     *   4. Then the new problem is, we have 'L' bboxes to do NMS:
     *      a. if 'L' is still too large (exceeds BARRIER_LAG), do step 1~4
     * again with correct data offsets. b. if 'L' is in a satified range for NMS
     * kernel, then simply call the NMS kernel. c. if 'L' is 0, or we already
     * got expected 'out_top_k', then we are good to return.
     *
     * Importatn Notes:
     *   Since we have sorted the bboxes according to its scores previously,
     *   Thus, all the bboxes are ranked from the "Most Possible" to "Leas
     * Possible". This indicates that, the front 'M' bboxes is likely to
     * overlaps with left ['N' - 'M'] bboxes,
     *
     *   Therefore, in practice, the NMS Filter method proved to be very
     * efficient for lowering the problem size (number of bounding boxes to be
     * processed), and generates good performance.
     */
    activeKernelsDoCount
        [n_elem -
         1]<<<GS_mask, BS_mask, BS_mask * n_elem * sizeof(bool), stream>>>(
            MASK_SIZE, out_top_k - total_kept_count, nms_threshold, bboxes,
            indexes + p_offset, out_indexes + total_kept_count, kept_count);

    int* kept_count_hst;
    LOC_HST(kept_count_hst, sizeof(int));
    COPY_D2H_ASYNC(kept_count_hst, kept_count, sizeof(int), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor filter_kept_tensor;
    auto s = ctx->allocate_temp(DT_INT32,
                                TensorShape({static_cast<int64>(sizeof(int))}),
                                &filter_kept_tensor);
    if (!s.ok()) {
      std::cerr << "Failed to allocate filter_kept_tensor with size of " << size
                << std::endl;
      return 0;
    }
    int* filter_kept = filter_kept_tensor.flat<int>().data();
    MEMSET_DEV(filter_kept, 0, sizeof(int));  // set filter_kept 0 at init.

    total_kept_count += (*kept_count_hst);
    p_size = p_size - MASK_SIZE;

#ifdef PRINT_DEBUG
    printf("[DEBUG] front 1024 kept: %d\n", (*kept_count_hst));
    printf("[DEBUG] total kept: %d\n", total_kept_count);
    printf("[DEBUG] num of boxes to be filtered: %d\n", p_size);
#endif  // PRINT_DEBUG

    if (total_kept_count >= out_top_k) {
      CUDA_CHECK(cudaFreeHost(kept_count_hst));
      break;
    }

    p_offset += MASK_SIZE;

#if GOOGLE_CUDA
    CudaLaunchConfig config = GetCudaLaunchConfig(p_size, d);
    nmsFilter<<<config.block_count, config.thread_per_block,
                (*kept_count_hst) * sizeof(float4), stream>>>
#else
    const int BS_filter = BLOCK_MID;
    const int GS_filter = DIV_UP(p_size, BS_filter);
    nmsFilter<<<BS_filter, GS_filter, (*kept_count_hst) * sizeof(float4),
                stream>>>
#endif  // GOOGLE_CUDA
        (p_size, (*kept_count_hst),
         out_indexes + (total_kept_count - (*kept_count_hst)), nms_threshold,
         bboxes, indexes + p_offset, filter_kept);

    int* filter_kept_hst;
    LOC_HST(filter_kept_hst, sizeof(int));
    COPY_D2H_ASYNC(filter_kept_hst, filter_kept, sizeof(int), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /** last round filter filtered out all the bounding boxes **/
    if ((*filter_kept_hst) == 0) {
      p_size = 0;
      CUDA_CHECK(cudaFreeHost(kept_count_hst));
      CUDA_CHECK(cudaFreeHost(filter_kept_hst));
      break;
    }

    int* temp_indexes_hst;
    LOC_HST(temp_indexes_hst, (p_size) * sizeof(int));
    COPY_D2H_ASYNC(temp_indexes_hst, indexes + p_offset, (p_size) * sizeof(int),
                   stream);
    int* temp_filter_kept_indexes;
    LOC_HST(temp_filter_kept_indexes, (*filter_kept_hst) * sizeof(int));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int ct = 0;
    for (int i = 0; i < (p_size); ++i) {
      const int kept_idx = temp_indexes_hst[i];
      if (kept_idx != -1) {
        temp_filter_kept_indexes[ct++] = kept_idx;
      }
    }

#ifdef PRINT_DEBUG
    printf("[DEBUG] before filter num of boxes: %d\n", p_size);
    printf("[DEBUG] after filter kept: %d\n", (*filter_kept_hst));
#endif  // PRINT_DEBUG
    p_size = (*filter_kept_hst);
    COPY_H2D_ASYNC(indexes + p_offset, temp_filter_kept_indexes,
                   (*filter_kept_hst) * sizeof(int), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef PRINT_DEBUG
    printf("[DEBUG] after filter num of boxes: %d\n", p_size);
    printf("--------------------------------\n");
#endif  // PRINT_DEBUG

    /** free temprary used memories **/
    CUDA_CHECK(cudaFreeHost(kept_count_hst));
    CUDA_CHECK(cudaFreeHost(filter_kept_hst));
    CUDA_CHECK(cudaFreeHost(temp_indexes_hst));
    CUDA_CHECK(cudaFreeHost(temp_filter_kept_indexes));
  }

  /** last round filter filtered out all the bounding boxes **/
  if (p_size == 0) {
#ifdef PRINT_DEBUG
    printf("[DEBUG] total kept bboxes num: %d, expect top k: %d\n",
           total_kept_count, out_top_k);
#endif  // PRINT_DEBUG
    return total_kept_count;
  }

  if (total_kept_count < out_top_k) {
    if (p_size > BARRIER_SML) {
#ifdef PRINT_DEBUG
      printf("[DEBUG] nms with size %d (kernel for range %d to %d)\n", size,
             BARRIER_SML, BARRIER_LAG);
#endif  // PRINT_DEBUG
      int last_group_kept = nmsGpuMediumSize(
          stream, p_size, out_top_k - total_kept_count, nms_threshold, bboxes,
          indexes + p_offset, out_indexes + total_kept_count, kept_count);
      total_kept_count += last_group_kept;
    } else {
#ifdef PRINT_DEBUG
      printf("[DEBUG] nms with size %d (kernel for size less than %d)\n", size,
             BARRIER_SML);
#endif  // PRINT_DEBUG
      int last_group_kept = nmsGpuSmallSize(
          stream, p_size, out_top_k - total_kept_count, nms_threshold, bboxes,
          indexes + p_offset, out_indexes + total_kept_count, kept_count);
      total_kept_count += last_group_kept;
    }
  }

#ifdef PRINT_DEBUG
  printf("[DEBUG] total_kept_count: %d, expect top k: %d\n", total_kept_count,
         out_top_k);
#endif  // PRINT_DEBUG

  return total_kept_count;
}

int nmsEngine(OpKernelContext* ctx, cudaStream_t stream,
#if GOOGLE_CUDA
              const GPUDevice& d,
#endif  // GOOGLE_CUDA
              const int size, const int out_top_k, const float nms_threshold,
              const float* bboxes, const float* scores, int* out_indexes,
              const bool do_filter_score = false,
              const float score_threshold = FLT_MAX) {
  /* Stage 1:
   *   init a index array pointing to the bounding boxes.
   *   Sort [score, bbox-index] pair by scores in descending order.
   *
   *   NOTE:
   *     We are involving indexing here, considering that 'None Max Suppression'
   *     is likely to be working on Sparse Storage Bounding Boxes.
   */
  Tensor sorted_indexes_tensor;
  auto s = ctx->allocate_temp(
      DT_INT32, TensorShape({static_cast<int64>(size * sizeof(int))}),
      &sorted_indexes_tensor);
  if (!s.ok()) {
    std::cerr << "Failed to allocate sorted_indexes_tensor with size of "
              << size << std::endl;
    return 0;
  }
  int* sorted_indexes = sorted_indexes_tensor.flat<int>().data();

  /* if do filter socre, we need to count the kept boxes number after score
   * filter */
  int score_filtered_kept = 0;

  /*
   * kept_count:
   * Count of boxes kept number, after NMS kernel.
   *  Summed by several NMS kernel executions
   */
  Tensor kept_count_tensor;
  s = ctx->allocate_temp(DT_INT32,
                         TensorShape({static_cast<int64>(size * sizeof(int))}),
                         &kept_count_tensor);
  if (!s.ok()) {
    std::cerr << "Failed to allocate kept_count_tensor with size of " << size
              << std::endl;
    return 0;
  }
  int* kept_count = kept_count_tensor.flat<int>().data();
  MEMSET_DEV(kept_count, 0, sizeof(int));

  sortScoreIndexPairGpu(ctx, stream,
#if GOOGLE_CUDA
                        d,
#endif  // GOOGLE_CUDA
                        size, scores, sorted_indexes, &score_filtered_kept,
                        do_filter_score, score_threshold);

  /*
   * If we do filter score by a threshold,
   * we need to check the kept boxes number here, in case
   * all the boxes are filtered out by score threshold.
   */
  if (do_filter_score) {
#if PRINT_DEBUG
    printf("[DEBUG] after score filter kept boxes number: %d\n",
           score_filtered_kept);
#endif  // PRINT_DEBUG
    if ((score_filtered_kept) == 0) return 0;
  }

  /* Stage 2:
   *   None Maximum Suppression on sorted bounding boxes (ordered by descending
   * scores) Generally we have two types of NMS module:
   *   - typical NMS:
   *       activate when the number of bboxes within a
   *       satisfied range for the kernel implementation.
   *   - filter-then-NMS:
   *       activate when the number of bboxes is too large,
   *       therefore, we needs a filter to decrease the amount of bboxes.
   */
  int final_kept;
  if (size > BARRIER_LAG) {
#ifdef PRINT_DEBUG
    printf("[DEBUG] nms with size %d (kernel for size greater than %d)\n", size,
           BARRIER_LAG);
#endif  // PRINT_DEBUG
    final_kept = nmsGpuLargeSize(ctx, stream,
#if GOOGLE_CUDA
                                 d,
#endif  // GOOGLE_CUDA
                                 size, out_top_k, nms_threshold,
                                 reinterpret_cast<const float4*>(bboxes),
                                 sorted_indexes, out_indexes, kept_count);
  } else if (size > BARRIER_SML) {
#ifdef PRINT_DEBUG
    printf("[DEBUG] nms with size %d (kernel for range %d to %d)\n", size,
           BARRIER_SML, BARRIER_LAG);
#endif  // PRINT_DEBUG
    final_kept = nmsGpuMediumSize(stream, size, out_top_k, nms_threshold,
                                  reinterpret_cast<const float4*>(bboxes),
                                  sorted_indexes, out_indexes, kept_count);
  } else {
#ifdef PRINT_DEBUG
    printf("[DEBUG] nms with size %d (kernel for size less than %d)\n", size,
           BARRIER_SML);
#endif  // PRINT_DEBUG
    final_kept = nmsGpuSmallSize(stream, size, out_top_k, nms_threshold,
                                 reinterpret_cast<const float4*>(bboxes),
                                 sorted_indexes, out_indexes, kept_count);
  }

  CUDA_CHECK(cudaGetLastError());
  return final_kept;
}

#if GOOGLE_CUDA
}  // namespace tensorflow
#endif  // GOOGLE_CUDA

#endif  // __NONE_MAX_SUPPRESSION_GPU_KERNEL_IMPL__

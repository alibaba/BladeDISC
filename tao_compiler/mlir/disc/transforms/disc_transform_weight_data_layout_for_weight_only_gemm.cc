// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-transform-weight-data-layout-for-weight-only-quant"

namespace mlir {
namespace disc_ral {

namespace {

enum class QuantType { INT8_WEIGHT_ONLY, PACKED_INT4_WEIGHT_ONLY };

// Note: It is not easy to introduce Faster-Transformer into disc_compiler_main.
// So we copied the code about weight layout conversion in Faster-Transformer
// directly here.
int get_bits_in_quant_type(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::INT8_WEIGHT_ONLY:
      return 8;
    case QuantType::PACKED_INT4_WEIGHT_ONLY:
      return 4;
    default:
      return -1;
  }
}

void add_bias_and_interleave_int8s_inplace(uint8_t* uint8_tensor,
                                           const size_t num_elts) {
  // Step 2 will transform the layout of a 32-bit register in CUDA in order to
  // match the int4 layout. This has no performance benefit and is purely so
  // that int4 and int8 have the same layout. Pictorially, this does the
  // following: bit 32                                                      0
  //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

  for (size_t base = 0; base < num_elts; base += 4) {
    std::swap(uint8_tensor[base + 1], uint8_tensor[base + 2]);
  }
}

void interleave_column_major_tensor(uint8_t* interleaved_quantized_tensor,
                                    const uint8_t* quantized_tensor,
                                    const std::vector<size_t>& shape,
                                    QuantType quant_type) {
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int BITS_PER_ELT = get_bits_in_quant_type(quant_type);
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int rows_per_tile = 64;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

  const int num_vec_rows = num_rows / elts_in_int32;
  const int vec_rows_per_tile = rows_per_tile / elts_in_int32;
  const int interleave = 2;

  for (int expert = 0; expert < num_experts; ++expert) {
    const int64_t matrix_offset =
        expert * int64_t(num_vec_rows) * int64_t(num_cols);
    for (int read_col = 0; read_col < num_cols; ++read_col) {
      const int64_t write_col = read_col / interleave;
      for (int base_vec_row = 0; base_vec_row < num_vec_rows;
           base_vec_row += vec_rows_per_tile) {
        for (int vec_read_row = base_vec_row;
             vec_read_row <
             std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
             ++vec_read_row) {
          const int64_t vec_write_row =
              interleave * base_vec_row +
              vec_rows_per_tile * (read_col % interleave) +
              vec_read_row % vec_rows_per_tile;

          const int64_t read_offset =
              matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
          const int64_t write_offset =
              matrix_offset + int64_t(write_col) * num_vec_rows * interleave +
              vec_write_row;
          output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
        }
      }
    }
  }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a
// substantial amount of time to transpose leading to long preprocessing times.
// This seemed to be a big issue for relatively large models.
template <QuantType quant_type>
void subbyte_transpose_impl(uint8_t* transposed_quantized_tensor,
                            const uint8_t* quantized_tensor,
                            const std::vector<size_t>& shape) {
  const int bits_per_elt = get_bits_in_quant_type(quant_type);

  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t col_bytes = num_cols * bits_per_elt / 8;
  const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
  const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

  const uint8_t* input_byte_ptr =
      reinterpret_cast<const uint8_t*>(quantized_tensor);
  uint8_t* output_byte_ptr =
      reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

  static constexpr int ELTS_PER_BYTE =
      quant_type == QuantType::INT8_WEIGHT_ONLY ? 1 : 2;

  static constexpr int M_TILE_L1 = 64;
  static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
  uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

  static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

  const int num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
  const int num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

  for (size_t expert = 0; expert < num_experts; ++expert) {
    const size_t matrix_offset = expert * num_rows * col_bytes;
    for (size_t row_tile_start = 0; row_tile_start < num_rows;
         row_tile_start += M_TILE_L1) {
      for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes;
           col_tile_start_byte += N_TILE_L1) {
        const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
        const int col_limit =
            std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          const int row = row_tile_start + ii;

          for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
            const int col = col_tile_start_byte + jj;

            const size_t logical_src_offset =
                matrix_offset + row * col_bytes + col;

            if (row < row_limit && col < col_limit) {
              for (int v = 0; v < VECTOR_WIDTH; ++v) {
                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
              }
            }
          }
        }

        if (quant_type == QuantType::INT8_WEIGHT_ONLY) {
          for (int ii = 0; ii < M_TILE_L1; ++ii) {
            for (int jj = ii + 1; jj < N_TILE_L1; ++jj) {
              std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
              // std::cout << "stageB: " << ii << ", " << jj << std::endl;
            }
          }
        } else if (quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY) {
          for (int ii = 0; ii < M_TILE_L1; ++ii) {
            // Using M_TILE_L1 here is deliberate since we assume that the cache
            // tile is square in the number of elements (not necessarily the
            // number of bytes).
            for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
              const int ii_byte = ii / ELTS_PER_BYTE;
              const int ii_bit_offset = ii % ELTS_PER_BYTE;

              const int jj_byte = jj / ELTS_PER_BYTE;
              const int jj_bit_offset = jj % ELTS_PER_BYTE;

              uint8_t src_elt =
                  0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
              uint8_t tgt_elt =
                  0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

              cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
              cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

              cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
              cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
            }
          }
        } else {
          ;
        }

        const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
        const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

        const int row_limit_trans =
            std::min(row_tile_start_trans + M_TILE_L1, num_cols);
        const int col_limit_trans =
            std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          const int row = row_tile_start_trans + ii;
          for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
            const int col = col_tile_start_byte_trans + jj;

            const size_t logical_tgt_offset =
                matrix_offset + row * col_bytes_trans + col;

            if (row < row_limit_trans && col < col_limit_trans) {
              for (int v = 0; v < VECTOR_WIDTH; ++v) {
                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
              }
            }
          }
        }
      }
    }
  }
}

void permute_B_rows_for_mixed_gemm(uint8_t* permuted_quantized_tensor,
                                   const uint8_t* quantized_tensor,
                                   const std::vector<size_t>& shape,
                                   const int64_t arch_version) {
  // We only want to run this step for weight only quant.
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int BITS_PER_ELT = 8;  // TODO: support 4bit quantization
  const int K = 16 / BITS_PER_ELT;
  const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
  const int ELTS_PER_REG = 32 / BITS_PER_ELT;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

  int MMA_SHAPE_N = 8;
  int B_ROWS_PER_MMA = 8 * K;
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int num_vec_cols = num_cols / elts_in_int32;

  // The code is written as below so it works for both int8 and packed int4.
  for (int expert = 0; expert < num_experts; ++expert) {
    const int64_t matrix_offset =
        expert * int64_t(num_rows) * int64_t(num_vec_cols);
    for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
      for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {
        for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
          const int write_row = base_row + tile_row;
          const int tile_read_row = 8 * (((tile_row % ELTS_PER_REG) / 2)) +
                                    tile_row % 2 +
                                    2 * (tile_row / ELTS_PER_REG);
          const int read_row = base_row + tile_read_row;
          const int read_col = write_col;

          const int64_t read_offset =
              matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
          const int64_t write_offset =
              matrix_offset + int64_t(write_row) * num_vec_cols + write_col;
          output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
        }
      }
    }
  }
}

struct WeightLayoutConverter
    : public OpRewritePattern<mhlo_disc::CustomCallV2Op> {
  using OpRewritePattern<mhlo_disc::CustomCallV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::CustomCallV2Op op,
                                PatternRewriter& rewriter) const override {
    auto call_target_name = op.getCallTargetName();
    if (call_target_name != "ral_pdll_weight_only_qgemm") {
      return failure();
    }

    // weight_should_be_reordered is used to control whether the weight layout
    // should be converted.
    DictionaryAttr custom_attr = op.getCustomAttrs();
    bool has_target_attr = custom_attr.contains("weight_should_be_reordered");
    if (!has_target_attr) {
      return failure();
    }
    bool weight_should_be_reordered =
        custom_attr.getAs<BoolAttr>("weight_should_be_reordered").getValue();
    if (!weight_should_be_reordered) {
      return failure();
    }

    // weight should come from constant op
    auto weight_op =
        dyn_cast_or_null<mhlo::ConstantOp>(op->getOperand(1).getDefiningOp());
    if (!weight_op) {
      return failure();
    }

    // weight should be ui8 (e.g. unsigned int 8)
    auto weight_type = weight_op.getResult().getType().cast<RankedTensorType>();
    if (!weight_type.getElementType().isUnsignedInteger(8)) {
      return failure();
    }

    // only weights of rank 2 are supported
    ArrayRef<int64_t> weight_shape = weight_type.getShape();
    if (weight_shape.size() != 2) {
      return failure();
    }

    // TODO: support transpose
    // Convert the weights layout.
    // For the sake of simplicity, we first convert the data structure of mlir
    // into the data structure required by ft, then process the weight, and
    // finally convert the processed weight back to mlir
    int64_t k = weight_shape[0];
    int64_t n = weight_shape[1];
    const int64_t num_bytes = k * n;
    auto weight_value = weight_op.getValue().getValues<APInt>().begin();
    std::vector<uint8_t> src_buf(num_bytes);
    std::vector<uint8_t> dst_buf(num_bytes);
    for (int i = 0; i < num_bytes; i++) {
      src_buf[i] = uint8_t(static_cast<APInt>(weight_value[i]).roundToDouble());
    }
    std::vector<int64_t> weight_shape_int64_t = weight_shape.vec();
    std::vector<size_t> weight_shape_size_t(weight_shape.size());
    std::transform(weight_shape_int64_t.begin(), weight_shape_int64_t.end(),
                   weight_shape_size_t.begin(),
                   [](int64_t i) { return static_cast<size_t>(i); });
    permute_B_rows_for_mixed_gemm(dst_buf.data(), src_buf.data(),
                                  weight_shape_size_t, 80);
    src_buf.swap(dst_buf);
    subbyte_transpose_impl<QuantType::INT8_WEIGHT_ONLY>(
        dst_buf.data(), src_buf.data(), weight_shape_size_t);
    src_buf.swap(dst_buf);
    interleave_column_major_tensor(dst_buf.data(), src_buf.data(),
                                   weight_shape_size_t,
                                   QuantType::INT8_WEIGHT_ONLY);
    src_buf.swap(dst_buf);
    add_bias_and_interleave_int8s_inplace(src_buf.data(), k * n);

    SmallVector<APInt> layout_reordered_weight(n * k, APInt(8, 0, false));
    for (int i = 0; i < num_bytes; i++) {
      layout_reordered_weight[i] = APInt(8, src_buf[i], false);
    }

    Location loc = op.getLoc();
    auto reordered_weight_op = rewriter.create<mhlo::ConstantOp>(
        loc, weight_type,
        DenseElementsAttr::get(weight_type, layout_reordered_weight));
    // // create a new custom call op with weight_should_be_reordered = false
    auto ctx = rewriter.getContext();
    auto now_weight_should_be_reordered = rewriter.getNamedAttr(
        "weight_should_be_reordered", rewriter.getBoolAttr(false));
    std::vector<NamedAttribute> new_custom_attr_members_vec;
    for (auto c : custom_attr) {
      if (c.getName().str() == "weight_should_be_reordered") {
        new_custom_attr_members_vec.push_back(now_weight_should_be_reordered);
      } else {
        new_custom_attr_members_vec.push_back(c);
      }
    }
    ArrayRef<NamedAttribute> new_custom_attr_members(
        new_custom_attr_members_vec);
    DictionaryAttr new_custom_attr =
        rewriter.getDictionaryAttr(new_custom_attr_members);

    SmallVector<Value> new_operands;
    for (Value operand : op.getOperands()) {
      new_operands.push_back(operand);
    }
    new_operands[1] = reordered_weight_op->getResult(0);
    auto new_custom_call_v2_op = rewriter.create<mhlo_disc::CustomCallV2Op>(
        loc, op.getResultTypes(), new_operands, op.getCallTargetName(),
        new_custom_attr, op.getHasSideEffect(), op.getDevice(),
        op.getInputPlacements(), op.getOutputPlacements(), op.getInputLayouts(),
        op.getOutputLayouts(), op.getExpectedInputLayouts(),
        op.getExpectedOutputLayouts());
    rewriter.replaceOp(op, new_custom_call_v2_op.getResult(0));

    return success();
  }
};

void populateTransformWeightLayoutPatterns(RewritePatternSet& patterns) {
  patterns.insert<WeightLayoutConverter>(patterns.getContext());
}

struct DiscTranformWeightDataLayoutForWeightOnlyQuantPass
    : public DiscTranformWeightDataLayoutForWeightOnlyQuantPassBase<
          DiscTranformWeightDataLayoutForWeightOnlyQuantPass> {
  void runOnOperation() override;
};

void DiscTranformWeightDataLayoutForWeightOnlyQuantPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateTransformWeightLayoutPatterns(patterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscTranformWeightDataLayoutForWeightOnlyQuantPass() {
  return std::make_unique<DiscTranformWeightDataLayoutForWeightOnlyQuantPass>();
}

}  // namespace disc_ral
}  // namespace mlir
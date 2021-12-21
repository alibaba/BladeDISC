#ifndef TAO_TAO_BRIDGE_PASSES_TAO_BACE_REFORMAT_FP32_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_BACE_REFORMAT_FP32_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"

namespace tensorflow {
namespace tao {

class TaoBaCEReformatPass : public GraphOptimizationPass {
 public:
  // Constructor for normal pass run.
  TaoBaCEReformatPass();

  // Test-only constructor.
  TaoBaCEReformatPass(bool enabled, int64 max_dim_bar, int64 min_dim_bar,
                      int64 size_bar)
      : enabled_(enabled),
        max_dim_bar_(max_dim_bar),
        min_dim_bar_(min_dim_bar),
        size_bar_(size_bar) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  bool enabled_ = false;    // if enable this pass.
  int64 max_dim_bar_ = -1;  // Upper threashold for size of larger dim.
  int64 min_dim_bar_ = -1;  // Upper threashold for size of smaller dim.
  int64 size_bar_ = -1;     // Upper threashold for size const.
};

}  // namespace tao

}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_PASSES_BACE_REFORMAT_FP32_PASS_H_

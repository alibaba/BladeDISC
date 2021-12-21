#ifndef TAO_TAO_BRIDGE_PASSES_TAO_DEFUSE_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_DEFUSE_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

class TaoDefusePass : public GraphOptimizationPass {
 public:
  TaoDefusePass(bool use_tvm) : GraphOptimizationPass() { use_tvm_ = use_tvm; }
  Status Run(const GraphOptimizationPassOptions& options) override;

  void set_opts(const std::unique_ptr<TaoPassOptions>& opt) {
    if (opt) {
      use_tvm_ = opt->use_tvm;
    }
  }

 private:
  bool use_tvm_;
};

const std::unordered_set<string> FusedOpList = {"_FusedConv2D", "_FusedMatMul",
                                                "_FusedBatchNormEx"};
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_BUILD_TAO_OP_PASS_H_

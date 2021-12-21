#ifndef TAO_TAO_BRIDGE_PASSES_TAO_BUILD_TAO_OP_PASS_H_
#define TAO_TAO_BRIDGE_PASSES_TAO_BUILD_TAO_OP_PASS_H_

#include "tao_bridge/common.h"
#include "tao_bridge/passes/tao_optimization_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tao {

class TaoBuildTaoOpPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  void set_opts(const std::unique_ptr<TaoPassOptions>& opt) {
    if (opt) {
      inner_tao_launch_ = opt->inner_tao_launch;
      use_tvm_ = opt->use_tvm;
    }
  }

 private:
  bool use_tvm_;
  bool inner_tao_launch_{false};  // inner attribute of TaoLaunch op.
};

}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_BUILD_TAO_OP_PASS_H_


#include <string>
#include <vector>

namespace mlir_test {

enum class BackendType;

bool feature_test_main(const std::string& mlir_file_path,
                       const std::vector<BackendType>& backend_types,
                       int num_inputs, int num_outputs,
                       const std::vector<std::string>& input_descriptors,
                       const std::vector<std::string>& output_descriptors,
                       const std::vector<std::vector<float>>& input_vals = {},
                       bool profiling = false, bool multi_cc_mode = false,
                       bool multi_cc_mode_dbg_ptx_only = false);

}  // namespace mlir_test

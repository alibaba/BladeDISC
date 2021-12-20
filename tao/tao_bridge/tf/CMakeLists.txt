list(APPEND TF_PORT_HEADERS
    "const_analysis.h"
    "deadness_analysis.h"
    "deadness_analysis_internal.h"
    "dump_graph.h"
    "graphcycles.h"
    "mark_for_compilation_pass_flags.h"
    "parse_flags_from_env.h"
    "resource_operation_safety_analysis.h"
    "resource_operation_table.h"
    "shape_inference_helpers.h"
    "status.h"
    "statusor.h"
    "subprocess.h"
    "types.h"
    "union_find.h"
    "util.h"
    "xla_cluster_util.h"
    "xla_op_registry.h"
    "defs.h"
    "compilability_check_util.h"
    "xla_config_registry.h"
    "device_util.h"
    "flags.h"
    "tf2xla_util.h"
    "lower_if_op.h"
    "lower_while_op.h"
)

list(APPEND TF_PORT_SOURCES
    "const_analysis.cc"
    "deadness_analysis.cc"
    "dump_graph.cc"
    "graphcycles.cc"
    "mark_for_compilation_pass_flags.cc"
    "parse_flags_from_env.cc"
    "resource_operation_safety_analysis.cc"
    "resource_operation_table.cc"
    "shape_inference_helpers.cc"
    "subprocess.cc"
    "util.cc"
    "xla_cluster_util.cc"
    "xla_op_registry.cc"
    "defs.cc"
    "compilability_check_util.cc"
    "xla_config_registry.cc"
    "device_util.cc"
    "flags.cc"
    "tf2xla_util.cc"
    "register_tensor_array_ops.cc"
    "lower_if_op.cc"
    "lower_while_op.cc"
)

list(APPEND TF_PORT_TESTS
    "util_test.cc"
    "const_analysis_test.cc"
    "deadness_analysis_test.cc"
    "resource_operation_table_test.cc"
    "resource_operation_safety_analysis_test.cc"
    "xla_op_registry_test.cc"
    "xla_cluster_util_test.cc"
    "graphcycles_test.cc"
)

add_library(tf_port OBJECT ${TF_PORT_SOURCES})

tao_cc_test(
  NAME tf_port_tests
  SRCS ${TF_PORT_TESTS}
)

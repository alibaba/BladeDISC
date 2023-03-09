list(APPEND PASSES_HEADERS
    "tao_build_tao_op_pass.h"
    "tao_encapsulate_subgraphs_pass.h"
    "tao_mark_for_compilation_pass.h"
    "tao_optimization_pass.h"
    "tao_partially_decluster_pass.h"
    "tao_bace_reformat_pass.h"
    "tao_clone_constants_for_better_clustering.h"
    "tao_defuse_pass.h"
    "functionalize_control_flow.h"
    "functionalize_cond.h"
    "functionalize_while.h"
    "functionalize_control_flow_util.h"
    "defunctionalize_control_flow.h"
    "tao_feature_detector.h"
)

list(APPEND PASSES_SOURCES
    "tao_build_tao_op_pass.cc"
    "tao_encapsulate_subgraphs_pass.cc"
    "tao_mark_for_compilation_pass.cc"
    "tao_optimization_pass.cc"
    "tao_partially_decluster_pass.cc"
    "tao_bace_reformat_pass.cc"
    "tao_clone_constants_for_better_clustering.cc"
    "tao_defuse_pass.cc"
    "functionalize_control_flow.cc"
    "functionalize_cond.cc"
    "functionalize_while.cc"
    "functionalize_control_flow_util.cc"
    "defunctionalize_control_flow.cc"
    "tao_feature_detector.cc"
)

add_library(passes OBJECT ${PASSES_SOURCES})

list(APPEND PASSES_TESTS
    "tao_bace_reformat_pass_test.cc"
)

list(APPEND CPU_PASSES_TESTS
    "tao_clone_constants_for_better_clustering_test.cc"
    "tao_defuse_pass_test.cc"
)

tao_cc_test(
  NAME passes_tests
  SRCS ${PASSES_TESTS}
)

tao_cc_test(
  NAME cpu_passes_tests
  SRCS ${CPU_PASSES_TESTS}
)

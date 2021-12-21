#pragma once

#include "tensorflow/core/public/version.h"

// defs moved or renamed between headers in different tf versions

#if TF_MAJOR_VERSION > 1
// TF2.4
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/graph_node_util.h"
#else
// TF1.12, TF1.15
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#endif

#if TF_MAJOR_VERSION > 1 || TF_MINOR_VERSION > 12
// TF1.15, TF2.4
#include "tensorflow/core/framework/bounds_check.h"
#else
// TF1.12
#include "tensorflow/core/kernels/bounds_check.h"
#endif

// put these macros in the end for new version absl also defined them
#if TF_MAJOR_VERSION > 1
// TF2.4
#ifndef GUARDED_BY
#define GUARDED_BY TF_GUARDED_BY
#endif

#ifndef EXCLUSIVE_LOCKS_REQUIRED
#define EXCLUSIVE_LOCKS_REQUIRED TF_EXCLUSIVE_LOCKS_REQUIRED
#endif
#endif
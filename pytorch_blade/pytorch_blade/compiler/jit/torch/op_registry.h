/* From PyTorch:
- *
- * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
- * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
- * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
- * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
- * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
- * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
- * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon
- * Bottou, Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research
- * Institute (Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute
- * (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
- */

#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <memory>
#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/compiler/jit/torch/schema_set.h"

namespace torch {
namespace blade {

// Moved from shape_analysis.cpp

// Requirements:
//   dims           : preserved from the first argument
//   scalar type    : preserved from the first argument (doesn't have to
//                    match other arguments)
//   device         : always matching and preserved
//   tensor inputs  : *
//   tensor outputs : 1
// NB: those ops (with slight adjustments) are good candidates for restarts.
//     Knowing the type and device of weights or biases is usually enough to
//     infer the output type.
std::shared_ptr<SchemaSet> nn_ops_first_input_preserving();

// Requirements:
//   dims           : Changed from first argument
//   scalar type    : preserved from the first argument
//   device         : always matching and preserved
//   tensor inputs  : 1
//   tensor outputs : 1
std::shared_ptr<SchemaSet> ops_one_tensor_in_shape_transform();
} // namespace blade
} // namespace torch

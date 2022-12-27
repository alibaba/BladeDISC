#include <iostream>  // Only for debugging purpose.
#include <vector>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"

// This file maintains the headers required by the schedule of CUTLASS-based
// compute-intensive op code generation. The headers will be preprocessed
// separated with the main body of codegen schedules.
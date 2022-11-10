// RUN: torch-disc-pdll --payload-input %s --pdl-input %p/conv_relu.pdll  | FileCheck %s

// CHECK-LABEL: @matched
func.func @matched(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %none, %0, %1, %2, %int1 : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

// CHECK-LABEL: func.func @not_matched_due_to_bias_not_none
func.func @not_matched_due_to_bias_not_none(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %bias: !torch.vtensor) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.conv2d
  // CHECK: torch.aten.relu
  // CHECK-NOT: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %bias, %0, %1, %2, %int1 : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

// CHECK-LABEL: func.func @not_matched_due_to_stride_not_const
func.func @not_matched_due_to_stride_not_const(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %stride: !torch.list<int>) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.conv2d
  // CHECK: torch.aten.relu
  // CHECK-NOT: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %none, %stride, %1, %2, %int1 : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

// CHECK-LABEL: func.func @not_matched_due_to_padding_not_const
func.func @not_matched_due_to_padding_not_const(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %padding: !torch.list<int>) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.conv2d
  // CHECK: torch.aten.relu
  // CHECK-NOT: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %none, %0, %padding, %2, %int1 : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

// CHECK-LABEL: func.func @not_matched_due_to_output_padding_not_const
func.func @not_matched_due_to_output_padding_not_const(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %output_padding: !torch.list<int>) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.conv2d
  // CHECK: torch.aten.relu
  // CHECK-NOT: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %none, %0, %1, %output_padding, %int1 : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

// CHECK-LABEL: func.func @not_matched_due_to_groups_not_const
func.func @not_matched_due_to_groups_not_const(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %groups: !torch.int) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.aten.conv2d
  // CHECK: torch.aten.relu
  // CHECK-NOT: torch.operator
  %3 = torch.aten.conv2d %arg0, %arg1, %none, %0, %1, %2, %groups : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
  %4 = torch.aten.relu %3 : !torch.vtensor -> !torch.vtensor
  return %4 : !torch.vtensor
}

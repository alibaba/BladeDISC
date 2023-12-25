// RUN: disc-opt -disc-input-output-alias \
// RUN: %s -o - | FileCheck %s

// CHECK-LABEL: main
func.func @main(%arg0: tensor<200x200xf32>, %arg1: tensor<200x200xf32>) -> (tensor<200x200xf32>, tensor<200x200xf32>) attributes {tf.entry_function = {input_output_alias_outputs = "0,1", input_output_alias_params = "0,1", input_placements = "gpu,gpu", output_placements = "gpu,gpu"}} {
    // CHECK: %0 = mhlo.add %arg1, %arg0 : tensor<200x200xf32>    
    %0 = mhlo.add %arg1, %arg0 : tensor<200x200xf32>
    // CHECK: "mhlo_disc.args_mutation"(%0, %arg1) : (tensor<200x200xf32>, tensor<200x200xf32>) -> ()
    // CHECK: return %arg0, %0 : tensor<200x200xf32>, tensor<200x200xf32>
    return %arg0, %0 : tensor<200x200xf32>, tensor<200x200xf32>
}

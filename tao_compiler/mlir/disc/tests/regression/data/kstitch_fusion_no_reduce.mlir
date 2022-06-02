// This UT has has two outputs with different number of elements. There is no
// reduce in this case. It should be fused into one kernel with kStitch type.
// This is static shape because I have no idea how to write dynamic-shaped one
// with TF dialect while telling the shape analyzer about the shape equality...
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func @main(%arg0: tensor<3797xf32>) -> (tensor<3797xf32>, tensor<3797x734xf32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:2 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg0) : (tensor<3797xf32>, tensor<3797xf32>) -> tensor<3797xf32>
      %1:2 = tf_executor.island wraps "tf.Mul"(%arg0, %arg0) : (tensor<3797xf32>, tensor<3797xf32>) -> tensor<3797xf32>
      %2:2 = tf_executor.island wraps "tf.Div"(%1, %0) : (tensor<3797xf32>, tensor<3797xf32>) -> tensor<3797xf32>
      %3:2 = tf_executor.island wraps "tf.Abs"(%2) : (tensor<3797xf32>) -> tensor<3797xf32>
      %c1:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %4:2 = tf_executor.island wraps "tf.ExpandDims"(%3, %c1) : (tensor<3797xf32>, tensor<i32>) -> (tensor<3797x1xf32>)
      %shape:2 = tf_executor.island wraps "tf.Const"() {value = dense<[3797, 734]> : tensor<2xi32>} : () -> tensor<2xi32>
      %7:2 = tf_executor.island wraps "tf.BroadcastTo"(%4, %shape) : (tensor<3797x1xf32>, tensor<2xi32>) -> tensor<3797x734xf32>
      %8:2 = tf_executor.island wraps "tf.Add"(%7, %7) : (tensor<3797x734xf32>, tensor<3797x734xf32>) -> tensor<3797x734xf32>
      %9:2 = tf_executor.island wraps "tf.Mul"(%7, %8) : (tensor<3797x734xf32>, tensor<3797x734xf32>) -> tensor<3797x734xf32>
      %10:2 = tf_executor.island wraps "tf.Sub"(%9, %8) : (tensor<3797x734xf32>, tensor<3797x734xf32>) -> tensor<3797x734xf32>
      tf_executor.fetch %3, %10 : tensor<3797xf32>, tensor<3797x734xf32>
    }
    return %graph#0, %graph#1 : tensor<3797xf32>, tensor<3797x734xf32>
  }
}
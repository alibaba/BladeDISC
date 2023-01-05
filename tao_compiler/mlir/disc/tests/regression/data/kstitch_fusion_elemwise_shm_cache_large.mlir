module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<96x512x512xf16>, %arg1: tensor<8x12x512x512xf16>, %arg2: tensor<8x12x512x512xf32>)
        -> (tensor<8x12x512x512xf16>, tensor<8x12x512x1xf32>, tensor<8x12x512x1xf32>, tensor<96x512x512xf16>)
        attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:4 = tf_executor.graph {
      %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<[96, 512, 512]> : tensor<3xi32>} : () -> tensor<3xi32>
      %3:2 = tf_executor.island wraps "tf.Const"() {value = dense<[8, 12, 512, 512]> : tensor<4xi32>} : () -> tensor<4xi32>
      %4:2 = tf_executor.island wraps "tf.Reshape"(%arg0, %3) : (tensor<96x512x512xf16>, tensor<4xi32>) -> tensor<8x12x512x512xf16>
      %5:2 = tf_executor.island wraps "tf.Div"(%4, %arg1) : (tensor<8x12x512x512xf16>, tensor<8x12x512x512xf16>) -> tensor<8x12x512x512xf16>
      %6:2 = tf_executor.island wraps "tf.Cast"(%5) : (tensor<8x12x512x512xf16>) -> (tensor<8x12x512x512xf32>)
      %9:2 = tf_executor.island wraps "tf.Add"(%6, %arg2) : (tensor<8x12x512x512xf32>, tensor<8x12x512x512xf32>) -> tensor<8x12x512x512xf32>
      %10:2 = tf_executor.island wraps "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
      %11:2 = tf_executor.island wraps "tf.Max"(%9, %10) : (tensor<8x12x512x512xf32>, tensor<i32>) -> tensor<8x12x512xf32>
      // return
      %12:2 = tf_executor.island wraps "tf.ExpandDims"(%11, %10) : (tensor<8x12x512xf32>, tensor<i32>) -> (tensor<8x12x512x1xf32>)
      %13:2 = tf_executor.island wraps "tf.BroadcastTo"(%12, %3) : (tensor<8x12x512x1xf32>, tensor<4xi32>) -> tensor<8x12x512x512xf32>
      %14:2 = tf_executor.island wraps "tf.Sub"(%9, %13) : (tensor<8x12x512x512xf32>, tensor<8x12x512x512xf32>) -> tensor<8x12x512x512xf32>
      %15:2 = tf_executor.island wraps "tf.Exp"(%14) : (tensor<8x12x512x512xf32>) -> tensor<8x12x512x512xf32>
      %16:2 = tf_executor.island wraps "tf.Sum"(%15, %10) : (tensor<8x12x512x512xf32>, tensor<i32>) -> tensor<8x12x512xf32>
      // return
      %17:2 = tf_executor.island wraps "tf.ExpandDims"(%16, %10) : (tensor<8x12x512xf32>, tensor<i32>) -> (tensor<8x12x512x1xf32>)
      %18:2 = tf_executor.island wraps "tf.BroadcastTo"(%17, %3) : (tensor<8x12x512x1xf32>, tensor<4xi32>) -> tensor<8x12x512x512xf32>
      %19:2 = tf_executor.island wraps "tf.Div"(%15, %18) : (tensor<8x12x512x512xf32>, tensor<8x12x512x512xf32>) -> tensor<8x12x512x512xf32>
      %20:2 = tf_executor.island wraps "tf.Cast"(%19) : (tensor<8x12x512x512xf32>) -> (tensor<8x12x512x512xf16>)
      %21:2 = tf_executor.island wraps "tf.Reshape"(%20, %2) : (tensor<8x12x512x512xf16>, tensor<3xi32>) -> tensor<96x512x512xf16>
      tf_executor.fetch %4, %12, %17, %21 : tensor<8x12x512x512xf16>, tensor<8x12x512x1xf32>, tensor<8x12x512x1xf32>, tensor<96x512x512xf16>
    }
    return %graph#0, %graph#1, %graph#2, %graph#3 : tensor<8x12x512x512xf16>, tensor<8x12x512x1xf32>, tensor<8x12x512x1xf32>, tensor<96x512x512xf16>
  }
}
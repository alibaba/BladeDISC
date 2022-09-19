module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg4: tensor<i32>, %arg9: tensor<?xi32>, %arg10: tensor<f32>, %arg11: tensor<?xf32>, %arg19: tensor<?x?x?xf32>) -> (tensor<?xi32>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:1 = tf_executor.graph {
      %cst:2 = tf_executor.island wraps "tf.Const"() {value = dense<[1, -1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %cst_0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %cst_1:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %cst_2:2 = tf_executor.island wraps "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %cst_3:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      %cst_4:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
      %cst_5:2 = tf_executor.island wraps "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
      %cst_6:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %cst_7:2 = tf_executor.island wraps "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
      %cst_8:2 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
      %0:2 = tf_executor.island wraps "tf.AddV2"(%arg9, %cst_1) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
      %1:2 = tf_executor.island wraps "tf.Mul"(%arg10, %arg19) {_XlaAlreadyClustered = true} : (tensor<f32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %2:2 = tf_executor.island wraps "tf.Shape"(%1) {_XlaAlreadyClustered = true} : (tensor<?x?x?xf32>) -> tensor<3xi32>
      %3:2 = tf_executor.island wraps "tf.StridedSlice"(%2, %cst_4, %cst_0, %cst_0) {_XlaAlreadyClustered = true, begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %4:2 = tf_executor.island wraps "tf.Pack"(%3, %cst_1, %cst_1) {_XlaAlreadyClustered = true, axis = 0 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
      %5:2 = tf_executor.island wraps "tf.StridedSlice"(%2, %cst_0, %cst_7, %cst_0) {_XlaAlreadyClustered = true, begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %6:2 = tf_executor.island wraps "tf.Range"(%arg4, %5, %cst_1) {_XlaAlreadyClustered = true} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      %7:2 = tf_executor.island wraps "tf.AddV2"(%6, %cst_1) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
      %8:2 = tf_executor.island wraps "tf.ExpandDims"(%7, %cst_6) {_XlaAlreadyClustered = true} : (tensor<?xi32>, tensor<i32>) -> tensor<1x?xi32>
      %9:2 = tf_executor.island wraps "tf.Cast"(%8) {Truncate = false, _XlaAlreadyClustered = true} : (tensor<1x?xi32>) -> tensor<1x?xf32>
      %10:2 = tf_executor.island wraps "tf.ExpandDims"(%0, %cst_5) {_XlaAlreadyClustered = true} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
      %11:2 = tf_executor.island wraps "tf.ExpandDims"(%arg9, %cst_5) {_XlaAlreadyClustered = true} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
      %12:2 = tf_executor.island wraps "tf.ExpandDims"(%9, %cst_5) {_XlaAlreadyClustered = true} : (tensor<1x?xf32>, tensor<i32>) -> tensor<1x?x1xf32>
      %13:2 = tf_executor.island wraps "tf.Reshape"(%arg11, %cst) {_XlaAlreadyClustered = true} : (tensor<?xf32>, tensor<2xi32>) -> tensor<1x?xf32>
      %14:2 = tf_executor.island wraps "tf.ExpandDims"(%13, %cst_1) {_XlaAlreadyClustered = true} : (tensor<1x?xf32>, tensor<i32>) -> tensor<1x1x?xf32>
      %15:2 = tf_executor.island wraps "tf.Mul"(%12, %14) {_XlaAlreadyClustered = true} : (tensor<1x?x1xf32>, tensor<1x1x?xf32>) -> tensor<1x?x?xf32>
      %16:2 = tf_executor.island wraps "tf.Cos"(%15) {_XlaAlreadyClustered = true} : (tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
      %17:2 = tf_executor.island wraps "tf.Sin"(%15) {_XlaAlreadyClustered = true} : (tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
      %18:2 = tf_executor.island wraps "tf.ConcatV2"(%17, %16, %cst_3) {_XlaAlreadyClustered = true} : (tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<i32>) -> tensor<1x?x?xf32>
      %19:2 = tf_executor.island wraps "tf.Tile"(%18, %4) {_XlaAlreadyClustered = true} : (tensor<1x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
      %20:2 = tf_executor.island wraps "tf.AddN"(%19, %1) {_XlaAlreadyClustered = true} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %21:2 = tf_executor.island wraps "tf.Shape"(%20) {_XlaAlreadyClustered = true} : (tensor<?x?x?xf32>) -> tensor<3xi32>
      %22:2 = tf_executor.island wraps "tf.StridedSlice"(%21, %cst_0, %cst_7, %cst_0) {_XlaAlreadyClustered = true, begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %23:2 = tf_executor.island wraps "tf.Range"(%arg4, %22, %cst_1) {_XlaAlreadyClustered = true} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      tf_executor.fetch %23 : tensor<?xi32>
    }
    return %graph : tensor<?xi32>
  }
}
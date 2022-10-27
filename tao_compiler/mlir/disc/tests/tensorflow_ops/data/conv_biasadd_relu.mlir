module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<?x?x?x?xf16>) -> (tensor<?x?x?x64xf16>) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}", input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph = tf_executor.graph {
      %weight:2 = tf_executor.island wraps "tf.Const"() {value = dense<-0.8> : tensor<1x1x64x64xf16>} : () -> tensor<1x1x64x64xf16>
      %const_0:2 = tf_executor.island wraps "tf.Const"() {value = dense<-1.0> : tensor<64xf16>} : () -> (tensor<64xf16>)
      %0:2 = tf_executor.island wraps "tf.Conv2D"(%arg0, %weight)
      {
        data_format = "NHWC",
        dilations = [1, 1, 1, 1],
        padding = "SAME",
        strides = [1, 1, 1, 1]
      } : (tensor<?x?x?x?xf16>, tensor<1x1x64x64xf16>) -> tensor<?x?x?x64xf16>
      %1:2 = tf_executor.island wraps "tf.BiasAdd"(%0, %const_0) {data_format = "NHWC"} : (tensor<?x?x?x64xf16>, tensor<64xf16>) -> tensor<?x?x?x64xf16>
      %2:2 = tf_executor.island wraps "tf.Relu"(%1) : (tensor<?x?x?x64xf16>) -> tensor<?x?x?x64xf16>
      tf_executor.fetch %2 : tensor<?x?x?x64xf16>
    }
    return %graph : tensor<?x?x?x64xf16>
  }
}
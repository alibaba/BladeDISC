module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(
        %arg00: tensor<768x768xf16>, %arg01: tensor<768x768xf16>, %arg02: tensor<768x768xf16>,
        %arg10: tensor<768x768xf16>, %arg11: tensor<768x768xf16>, %arg12: tensor<768x768xf16>,
        %arg20: tensor<768x768xf16>, %arg21: tensor<768x768xf16>, %arg22: tensor<768x768xf16>,
        %arg30: tensor<768x768xf16>, %arg31: tensor<768x768xf16>, %arg32: tensor<768x768xf16>,
        %arg40: tensor<768x768xf16>, %arg41: tensor<768x768xf16>, %arg42: tensor<768x768xf16>,
        %arg50: tensor<768x768xf16>, %arg51: tensor<768x768xf16>, %arg52: tensor<768x768xf16>,
        %arg60: tensor<768x768xf16>, %arg61: tensor<768x768xf16>, %arg62: tensor<768x768xf16>,
        %arg70: tensor<768x768xf16>, %arg71: tensor<768x768xf16>, %arg72: tensor<768x768xf16>,
        %arg80: tensor<768x768xf16>, %arg81: tensor<768x768xf16>, %arg82: tensor<768x768xf16>,
        %arg90: tensor<768x768xf16>, %arg91: tensor<768x768xf16>, %arg92: tensor<768x768xf16>,
        %arg100: tensor<768x768xf16>, %arg101: tensor<768x768xf16>, %arg102: tensor<768x768xf16>,
        %arg110: tensor<768x768xf16>, %arg111: tensor<768x768xf16>, %arg112: tensor<768x768xf16>
        ) -> (
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>,
            tensor<768x2304xf16>
        ) attributes {tf.entry_function = {inputs = "{{INPUTS}}", outputs = "{{OUTPUTS}}",
                                           input_placements="{{INPUT_PLACEMENTS}}", output_placements="{{OUTPUT_PLACEMENTS}}"}} {
    %graph:12 = tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %1:2 = tf_executor.island wraps "tf.ConcatV2"(%arg00, %arg01, %arg02, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %2:2 = tf_executor.island wraps "tf.ConcatV2"(%arg10, %arg11, %arg12, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %3:2 = tf_executor.island wraps "tf.ConcatV2"(%arg20, %arg21, %arg22, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %4:2 = tf_executor.island wraps "tf.ConcatV2"(%arg30, %arg31, %arg32, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %5:2 = tf_executor.island wraps "tf.ConcatV2"(%arg40, %arg41, %arg42, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %6:2 = tf_executor.island wraps "tf.ConcatV2"(%arg50, %arg51, %arg52, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %7:2 = tf_executor.island wraps "tf.ConcatV2"(%arg60, %arg61, %arg62, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %8:2 = tf_executor.island wraps "tf.ConcatV2"(%arg70, %arg71, %arg72, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %9:2 = tf_executor.island wraps "tf.ConcatV2"(%arg80, %arg81, %arg82, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %10:2 = tf_executor.island wraps "tf.ConcatV2"(%arg90, %arg91, %arg92, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %11:2 = tf_executor.island wraps "tf.ConcatV2"(%arg100, %arg101, %arg102, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      %12:2 = tf_executor.island wraps "tf.ConcatV2"(%arg110, %arg111, %arg112, %0) : (tensor<768x768xf16>, tensor<768x768xf16>, tensor<768x768xf16>, tensor<i32>) -> tensor<768x2304xf16>
      tf_executor.fetch 
          %1,  
          %2,  
          %3,  
          %4,  
          %5,  
          %6,  
          %7,  
          %8,  
          %9,  
          %10,  
          %11,  
          %12  
        : 
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>
    }
    return 
          %graph#0,
          %graph#1,
          %graph#2,
          %graph#3,
          %graph#4,
          %graph#5,
          %graph#6,
          %graph#7,
          %graph#8,
          %graph#9,
          %graph#10,
          %graph#11
        : 
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>,
          tensor<768x2304xf16>
  }
}

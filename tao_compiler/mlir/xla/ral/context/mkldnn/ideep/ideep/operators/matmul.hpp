#ifndef IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP

namespace ideep {

// Parameters for dynamic quantization
struct matmul_forward_dyn_quant_params {
  scale_t weight_scales;                 // to compute output scales
  tensor wei_zero_point_m;               // for matmul computation
  tensor::desc src_desc;                 // to create src tensor
  dnnl::reorder::primitive src_reorder;  // to reorder src

  matmul_forward_dyn_quant_params() {}

  matmul_forward_dyn_quant_params(scale_t&& weight_scales,
                                  tensor&& wei_zero_point_m,
                                  tensor::desc&& src_desc,
                                  dnnl::reorder::primitive&& src_reorder)
      : weight_scales(std::move(weight_scales)),
        wei_zero_point_m(std::move(wei_zero_point_m)),
        src_desc(std::move(src_desc)),
        src_reorder(std::move(src_reorder)) {}
};

// Common parameters for computation
struct matmul_forward_params {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul primitive;
  attr_t op_attr;
  attr_t src_attr;
  attr_t weights_attr;
  attr_t bias_attr;  // contains requantization scales for bias
  std::shared_ptr<matmul_forward_dyn_quant_params> dq_param_ptr;

  matmul_forward_params() {}

  matmul_forward_params(dnnl::matmul::primitive_desc&& pd, attr_t&& op_attr,
                        attr_t&& src_attr, attr_t&& weights_attr,
                        attr_t&& bias_attr)
      : pd(std::move(pd)),
        op_attr(std::move(op_attr)),
        src_attr(std::move(src_attr)),
        weights_attr(std::move(weights_attr)),
        bias_attr(std::move(bias_attr)) {
    primitive = dnnl::matmul(pd);
  }
};

struct matmul_forward : public dnnl::matmul,
                        utils::computation_cache<dnnl::matmul::primitive_desc> {
  using super = dnnl::matmul;

  template <bool keep_format = true, bool weight_format_any = false>
  static void compute(const tensor& src, const tensor& weights, tensor& dst,
                      const attr_t& attr = attr_t(),
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false, keep_format, weight_format_any>(
        src, weights, dummy_bias, dst, attr, aengine);
  }

  template <bool keep_format = true, bool weight_format_any = false>
  static primitive_desc get_primitive_desc(
      const tensor& src, const tensor& weights, tensor& dst,
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    src_desc = src.get_desc();
    weights_desc = weights.get_desc();
    dst_desc = dst.get_desc();
    return get_primitive_desc</*with_bias=*/false, keep_format,
                              weight_format_any>(
        src_desc, weights_desc, bias_desc, dst_desc, attr, aengine);
  }

  static void compute(super& primitive, const tensor& src,
                      const tensor& weights, tensor& dst,
                      const attr_t& attr = attr_t(),
                      const engine& aengine = engine::cpu_engine()) {
    primitive.execute(stream::default_stream(), {{DNNL_ARG_SRC, src},
                                                 {DNNL_ARG_WEIGHTS, weights},
                                                 {DNNL_ARG_DST, dst}});
  }

  static void compute(const tensor& src, const tensor& weights,
                      const tensor& bias, tensor& dst,
                      const float dst_coeff = 1.0f,
                      const float sum_coeff = 1.0f,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const data_type dst_type = data_type::undef,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst, dst_coeff, sum_coeff, src_scales,
        weights_scales, dst_scales, attr, dst_type, alowp_kind, aengine);
  }

  static void compute(const tensor& src, const tensor& weights, tensor& dst,
                      const float dst_coeff = 1.0f,
                      const float sum_coeff = 1.0f,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const data_type dst_type = data_type::undef,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst, dst_coeff, sum_coeff, src_scales,
        weights_scales, dst_scales, attr, dst_type, alowp_kind, aengine);
  }

  static tensor::desc expected_weights_desc(
      const dims& weights_dims, data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    auto ndims = weights_dims.size();
    dims x_dims = weights_dims;
    x_dims[ndims - 2] = 1;
    x_dims[ndims - 1] = weights_dims[ndims - 2];
    dims y_dims = {x_dims[0], weights_dims[1]};
    if (ndims == 3) y_dims = {x_dims[0], x_dims[1], weights_dims[2]};
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc x_desc(x_dims, x_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc y_desc(y_dims, y_dtype, ndims == 2 ? tag::ab : tag::abc);
    tensor::desc weights_desc(weights_dims, dtype,
                              ndims == 2 ? tag::ab : tag::abc);
    auto pd = primitive_desc({x_desc, weights_desc, y_desc}, aengine);
    return pd.weights_desc();
  }

  static tensor::desc expected_weights_desc(
      const tensor& src, const tensor& weights, tensor& dst,
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    auto weights_desc = weights.get_desc().to_format_any();
    auto dst_desc = dst.get_desc();
    auto pd = primitive_desc({src_desc, weights_desc, dst_desc}, aengine);
    return pd.weights_desc();
  }

  // Prepare for int8 op without bias.
  // Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
  template <bool is_dynamic = false>
  static void prepare(
      matmul_forward_params& param, const tensor& src, const tensor& weights,
      tensor& dst, const scale_t& src_scales, const scale_t& weights_scales,
      const scale_t& dst_scales, const zero_point_t& src_zero_points,
      const zero_point_t& dst_zero_points, const float dst_coeff = 1.0f,
      const float sum_coeff = 1.0f, const attr_t& attr = attr_t(),
      const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    if (is_dynamic) {
      do_prepare_dynamic_quant</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst, weights_scales, sum_coeff, attr,
          data_type::f32, aengine);
    } else {
      do_prepare_static_quant</*with_bias=*/false>(
          param, src, weights, dummy_bias, dst, src_scales, weights_scales,
          dst_scales, src_zero_points, dst_zero_points, dst_coeff, sum_coeff,
          attr, dst_type, alowp_kind, aengine);
    }
  }

  // Prepare for int8 op with bias.
  // Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
  template <bool is_dynamic = false>
  static void prepare(
      matmul_forward_params& param, const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst, const scale_t& src_scales,
      const scale_t& weights_scales, const scale_t& dst_scales,
      const zero_point_t& src_zero_points, const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f, const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(), const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    if (is_dynamic) {
      do_prepare_dynamic_quant</*with_bias=*/true>(
          param, src, weights, bias, dst, weights_scales, sum_coeff, attr,
          data_type::f32, aengine);
    } else {
      do_prepare_static_quant</*with_bias=*/true>(
          param, src, weights, bias, dst, src_scales, weights_scales,
          dst_scales, src_zero_points, dst_zero_points, dst_coeff, sum_coeff,
          attr, dst_type, alowp_kind, aengine);
    }
  }

  // Compute for fp32 and static int8 (int8 * int8 -> int8)
  // without bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const matmul_forward_params& param, const tensor& src,
                      const tensor& weights, tensor& dst) {
    static tensor dummy_bias;
    do_compute</*with_bias=*/false, reorder_src, reorder_weight>(
        param, src, weights, dummy_bias, dst);
  }

  // Compute for dynamic int8 (fp32 * int8 -> fp32)
  // Without bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_weight = true>
  static inline void compute(const matmul_forward_params& param,
                             const tensor& src, const tensor& weights,
                             tensor& dst, const scale_t& src_scales,
                             const zero_point_t& src_zero_points,
                             const float dst_coeff = 1.0f,
                             const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
        param, src, weights, dummy_bias, dst, src_scales, src_zero_points,
        dst_coeff, aengine);
  }

  // Compute for dynamic int8 (fp32 * int8 -> fp32)
  // With bias. Bias is not used if it is empty.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_weight = true>
  static inline void compute(const matmul_forward_params& param,
                             const tensor& src, const tensor& weights,
                             const tensor& bias, tensor& dst,
                             const scale_t& src_scales,
                             const zero_point_t& src_zero_points,
                             const float dst_coeff = 1.0f,
                             const engine& aengine = engine::cpu_engine()) {
    if (bias.is_empty()) {
      do_compute_dynamic_quant</*with_bias=*/false, reorder_weight>(
          param, src, weights, bias, dst, src_scales, src_zero_points,
          dst_coeff, aengine);
    } else {
      do_compute_dynamic_quant</*with_bias=*/true, reorder_weight>(
          param, src, weights, bias, dst, src_scales, src_zero_points,
          dst_coeff, aengine);
    }
  }

  // Compute for fp32 and static int8 (int8 * int8 -> int8)
  // with bias.
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool reorder_src = true, bool reorder_weight = true>
  static void compute(const matmul_forward_params& param, const tensor& src,
                      const tensor& weights, const tensor& bias, tensor& dst) {
    do_compute</*with_bias=*/true, reorder_src, reorder_weight>(
        param, src, weights, bias, dst);
  }

 private:
  template <bool with_bias = false, bool keep_format = true,
            bool weight_format_any = false>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc, const tensor::desc& weights_desc,
      const tensor::desc& bias_desc, const tensor::desc& dst_desc,
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_query = src_desc;
    auto weights_desc_query = weights_desc;
    auto bias_desc_query = with_bias ? bias_desc : tensor::desc();
    auto dst_desc_query = dst_desc;
    if (!keep_format) {
      src_desc_query = src_desc.to_format_any();
      weights_desc_query = weights_desc.to_format_any();
      bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
      dst_desc_query = dst_desc.to_format_any();
    } else if (weight_format_any) {
      weights_desc_query = weights_desc.to_format_any();
    }

    auto key = utils::create_key(src_desc_query, weights_desc_query, with_bias,
                                 keep_format, weight_format_any, attr);
    return fetch_or_create(key, [&]() {
      return with_bias ? primitive_desc({src_desc_query, weights_desc_query,
                                         bias_desc_query, dst_desc_query},
                                        attr, aengine)
                       : primitive_desc({src_desc_query, weights_desc_query,
                                         dst_desc_query},
                                        attr, aengine);
    });
  }

  template <bool with_bias = false, bool keep_format = true,
            bool weight_format_any = false>
  static void compute_impl(const tensor& src, const tensor& weights,
                           const tensor& bias, tensor& dst,
                           const attr_t& attr = attr_t(),
                           const engine& aengine = engine::cpu_engine()) {
    tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
    src_desc = src.get_desc();
    weights_desc = weights.get_desc();
    if (with_bias) {
      bias_desc = bias.get_desc();
    }
    dst_desc = dst.get_desc();
    auto pd = get_primitive_desc<with_bias, keep_format, weight_format_any>(
        src_desc, weights_desc, bias_desc, dst_desc, attr, aengine);
    if (with_bias) {
      super(pd).execute(stream::default_stream(), {{DNNL_ARG_SRC, src},
                                                   {DNNL_ARG_WEIGHTS, weights},
                                                   {DNNL_ARG_BIAS, bias},
                                                   {DNNL_ARG_DST, dst}});

    } else {
      super(pd).execute(stream::default_stream(), {{DNNL_ARG_SRC, src},
                                                   {DNNL_ARG_WEIGHTS, weights},
                                                   {DNNL_ARG_DST, dst}});
    }
  }

  // For dynamic int8 op (fp32 * int8 -> fp32)
  template <bool with_bias>
  static inline void do_prepare_dynamic_quant(
      matmul_forward_params& param, const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst, const scale_t& weights_scales,
      const float sum_coeff = 1.0f,  // for post-op sum
      const attr_t& attr = attr_t(), const data_type dst_type = data_type::f32,
      const engine& aengine = engine::cpu_engine()) {
    /* This function does the following things:
     * - Determine expected descs of src/weight/dst
     * - Use runtime values for op attributes
     * - Create matmul primitive desc and primitive
     * - Create reorder primitive for src (fp32 -> int8)
     */

    IDEEP_ENFORCE(src.ndims() == weights.ndims(),
                  "Invalid dims in src or weights");
    if (!param.dq_param_ptr) {
      param.dq_param_ptr = std::make_shared<matmul_forward_dyn_quant_params>();
    }
    IDEEP_ENFORCE(param.dq_param_ptr,
                  "Failed to allocate memory for parameters");

    tensor::desc& src_desc = param.dq_param_ptr->src_desc;
    attr_t& op_attr = param.op_attr;
    attr_t src_attr;

    tensor::dims src_dims = src.get_dims();
    tensor::dims dst_dims = {src_dims[0], weights.get_dim(1)};
    auto ndims = weights.ndims();
    if (ndims == 3)
      dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};

    auto& weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    auto src_data_type = data_type::u8;
    std::vector<int64_t> src_strides =
        (ndims == 3)
            ? std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1})
            : std::vector<int64_t>({src_dims[1], 1});
    src_desc = tensor::desc(src_dims, src_data_type, src_strides);

    // Prepare tensor of weight zero point
    static auto wei_zero_point = zero_point_t(1);
    const dim wei_zero_point_size = 1;
    const dim wei_zero_point_stride = 1;
    tensor::desc wei_zero_point_desc = {
        {wei_zero_point_size}, data_type::s32, {wei_zero_point_stride}};
    tensor wei_zero_point_m(wei_zero_point_desc,
                            reinterpret_cast<int*>(wei_zero_point.data()),
                            aengine);

    // Post-ops
    // For dynamic quantization, bias is applied by post-op add
    // so that overhead of bias reorder is avoided.
    // Need to 'prepend' post-op add to post op list.
    auto pops = attr.get_post_ops();
    dnnl::post_ops new_pops;
    if (with_bias) {
      new_pops.append_binary(dnnl::algorithm::binary_add, bias.get_desc());
    }
    for (int i = 0; i < pops.len(); ++i) {
      // Only sum and eltwise is supported now
      if (kind::sum == pops.kind(i)) {
        // The parameter sum_coeff is passed in explicitly now due to legacy
        // code. TO-DO: Remove the argument 'sum_coeff'. User should prepare all
        // post-ops in argument 'attr'.
        new_pops.append_sum(sum_coeff);
      } else if (kind::eltwise == pops.kind(i)) {
        float scale = 1.0, alpha = 1.0, beta = 0.0;
        dnnl::algorithm alg;
        pops.get_params_eltwise(i, scale, alg, alpha, beta);
        new_pops.append_eltwise(scale, alg, alpha, beta);
      }
    }
    op_attr.set_post_ops(new_pops);

    // fill primitive attr
    op_attr.set_output_scales(utils::op_scale_mask(1 /* scale_size */),
                              {DNNL_RUNTIME_F32_VAL});
    op_attr.set_zero_points(DNNL_ARG_SRC, utils::tensor_zp_mask(1),
                            {DNNL_RUNTIME_S32_VAL});
    op_attr.set_zero_points(DNNL_ARG_WEIGHTS, utils::tensor_zp_mask(1),
                            {DNNL_RUNTIME_S32_VAL});
    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Src attr for reorder
    src_attr.set_output_scales(utils::op_scale_mask(1), {DNNL_RUNTIME_F32_VAL});
    src_attr.set_zero_points(DNNL_ARG_DST, utils::tensor_zp_mask(1),
                             {DNNL_RUNTIME_S32_VAL});

    // Dst desc
    std::vector<int64_t> dst_strides =
        (ndims == 3)
            ? std::vector<int64_t>({dst_dims[2] * dst_dims[1], dst_dims[1], 1})
            : std::vector<int64_t>({dst_dims[1], 1});
    tensor::desc dst_desc = tensor::desc(dst_dims, dst_type, dst_strides);
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_type(dst_type);
    }

    // Create pd and primitive
    param.pd = primitive_desc({src_desc, weights.get_desc(), dst_desc}, op_attr,
                              aengine);
    param.primitive = super(param.pd);

    // Create src reorder primitive with runtime scales/zero point
    auto src_reorder_pd = dnnl::reorder::primitive_desc(
        aengine, src.get_desc(), aengine, src_desc, src_attr);
    param.dq_param_ptr->src_reorder = dnnl::reorder(src_reorder_pd);

    param.dq_param_ptr->weight_scales = std::move(weights_scales_in);
    param.dq_param_ptr->wei_zero_point_m = std::move(wei_zero_point_m);
  }

  template <bool with_bias>
  static void do_prepare_static_quant(
      matmul_forward_params& param, const tensor& src, const tensor& weights,
      const tensor& bias, tensor& dst, const scale_t& src_scales,
      const scale_t& weights_scales, const scale_t& dst_scales,
      const zero_point_t& src_zero_points, const zero_point_t& dst_zero_points,
      const float dst_coeff = 1.0f, const float sum_coeff = 1.0f,
      const attr_t& attr = attr_t(), const data_type dst_type = data_type::u8,
      const lowp_kind alowp_kind = u8s8,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(src.ndims() == weights.ndims(),
                  "Invalid dims in src or weights");
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t& op_attr = param.op_attr;
    attr_t& src_attr = param.src_attr;
    attr_t& weights_attr = param.weights_attr;
    attr_t& bias_attr = param.bias_attr;
    op_attr = attr;
    scale_t dst_scales_in;
    auto dst_data_type = data_type::u8;

    tensor::dims src_dims = src.get_dims();
    tensor::dims dst_dims = {src_dims[0], weights.get_dim(1)};
    auto ndims = weights.ndims();
    if (ndims == 3) {
      dst_dims = {src_dims[0], src.get_dim(1), weights.get_dim(2)};
    }

    auto& weights_scales_in =
        weights_scales.empty() ? IDEEP_DEF_SCALE : weights_scales;
    IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                  "Unsupported lowp kind");
    auto src_scales_in = src_scales.empty() ? IDEEP_DEF_SCALE : src_scales;
    auto src_data_type = (alowp_kind == u8s8) ? data_type::u8 : data_type::s8;
    std::vector<int64_t> src_strides =
        (ndims == 3)
            ? std::vector<int64_t>({src_dims[1] * src_dims[2], src_dims[1], 1})
            : std::vector<int64_t>({src_dims[1], 1});
    src_desc = tensor::desc(src_dims, src_data_type, tag::any);
    if (src.get_data_type() == data_type::f32) {
      src_attr = {0, src_scales_in};
    }

    int scale_size = (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
    weights_desc = weights.get_desc();
    if (weights.get_data_type() == data_type::f32) {
      weights_attr = {utils::tensor_scale_mask(scale_size, false),
                      weights_scales_in};
    }
    // determine dst data type
    if (dst.get_data_type() != data_type::undef) {
      dst_data_type = dst.get_data_type();
    } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
      dst_data_type = data_type::f32;
    } else {
      dst_data_type = data_type::u8;
    }

    // fill primitive attr
    scale_t op_scales(scale_size), bias_scales(scale_size);
    dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32)
                        ? IDEEP_DEF_SCALE
                        : dst_scales;
    const auto& src_zero_point =
        src.has_zero_point()
            ? src.get_zero_point()
            : src_zero_points.empty() ? IDEEP_DEF_ZP : src_zero_points;
    const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
    const auto& dst_zero_point =
        dst.has_zero_point()
            ? dst.get_zero_point()
            : dst_zero_points.empty() ? IDEEP_DEF_ZP : dst_zero_points;
    const auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
    IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1,
                  "DNNL only support 1-dim zero_point for activation");
    const auto& wei_zero_point =
        weights.has_zero_point() ? weights.get_zero_point() : IDEEP_DEF_ZP;

    if (attr.has_op_kind(kind::sum)) {
      float sum_scale = sum_coeff * dst_scales_in[0] /
                        (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
      op_attr = attr_t::fuse_sum(sum_scale);
    }

    auto bias_scales_in = bias.has_scale() ? bias.get_scale() : IDEEP_DEF_SCALE;
    bias_scales_in = bias_scales_in.size() == 1
                         ? std::vector<float>(scale_size, bias_scales_in[0])
                         : bias_scales_in;
    for (int i = 0; i < scale_size; i++) {
      bias_scales[i] = (dst_coeff * bias_scales_in[i]) /
                       (src_scales_in[0] * weights_scales_in[i]);
      op_scales[i] = (src_scales_in[0] * weights_scales_in[i]) /
                     (dst_coeff * dst_scales_in[0]);
    }
    op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);
    op_attr.set_zero_points(DNNL_ARG_SRC,
                            utils::tensor_zp_mask(src_zero_point.size()),
                            src_zero_point);
    if (src.get_data_type() == data_type::f32) {
      // Set zero point for src reorder (fp32 -> int8).
      // First arg should be DNNL_ARG_DST rather than DNNL_ARG_SRC
      src_attr.set_zero_points(DNNL_ARG_DST,
                               utils::tensor_zp_mask(src_zero_point.size()),
                               src_zero_point);
    }
    op_attr.set_zero_points(DNNL_ARG_WEIGHTS, utils::tensor_zp_mask(1),
                            zero_point_t(1, wei_zero_point[0]));
    if (dst_data_type != data_type::f32) {
      op_attr.set_zero_points(DNNL_ARG_DST,
                              utils::tensor_zp_mask(dst_zero_point.size()),
                              dst_zero_point);
    }

    if (with_bias) {
      tag bia_tag = bias.get_dims().size() == 2 ? tag::ab : tag::abc;
      bias_desc = {bias.get_dims(), data_type::s32,
                   bia_tag};  // Use f32 instead of s32 to improve accuracy
      if (bias.get_data_type() != data_type::s32) {
        auto ndims = bias.get_dims().size();
        int mask = scale_size > 1 ? 1 << (ndims - 1) : 0;
        bias_attr = {mask, bias_scales};
      }
    }

    op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;
    tensor::desc dst_desc = tensor::desc(dst_dims, dst_data_type, tag::any);
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_type(dst_data_type);
    }
    auto key = utils::create_key(src_desc, weights_desc, bias_desc, dst_desc,
                                 op_attr, with_bias, omp_get_max_threads());

    param.pd = fetch_or_create(key, [&]() {
      if (with_bias) {
        return primitive_desc({src_desc, weights_desc, bias_desc, dst_desc},
                              op_attr, aengine);
      } else {
        return primitive_desc({src_desc, weights_desc, dst_desc}, op_attr,
                              aengine);
      }
    });
    param.primitive = std::move(super(param.pd));
  }

  // For fp32 and static int8 op (int8 * int8 -> int8)
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_src, bool reorder_weight>
  static inline void do_compute(
      const matmul_forward_params& param, const tensor& src,
      const tensor& weights, const tensor& bias, tensor& dst,
      const std::vector<tensor>& bin_post_params = {}) {
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& op_attr = param.op_attr;
    auto& src_attr = param.src_attr;
    auto& weights_attr = param.weights_attr;
    auto& bias_attr = param.bias_attr;

    auto expected_src_desc = pd.src_desc();
    auto expected_wei_desc = pd.weights_desc();
    auto expected_dst_desc = pd.dst_desc();

    auto& expected_src =
        reorder_src ? src.reorder_if_differ_in(expected_src_desc, src_attr)
                    : src;
    auto& expected_weights =
        reorder_weight
            ? weights.reorder_if_differ_in(expected_wei_desc, weights_attr)
            : weights;
    tensor scratchpad(pd.scratchpad_desc());

    exec_args args;
    args.insert({DNNL_ARG_SRC, expected_src});
    args.insert({DNNL_ARG_WEIGHTS, expected_weights});
    args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
    auto& expected_bias =
        (with_bias && reorder_weight)
            ? bias.reorder_if_differ_in(pd.bias_desc(), bias_attr)
            : bias;
    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, expected_bias});
    }
    // Do not reorder these params. They may have different shapes as dst
    for (int i = 0; i < bin_post_params.size(); i++) {
      args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                   bin_post_params[i]});
    }
    if (reorder_src) {
      tensor expected_dst;
      if (dst.is_empty() || dst.get_desc() != expected_dst_desc) {
        // If dst buffer are not given by user or user given dst buffer are not
        // under expected format We need init a new one
        expected_dst.init(expected_dst_desc);
        if (!dst.is_empty() && op_attr.has_op_kind(kind::sum)) {
          // We need copy the content of given buffer if matmul is fused with
          // sum
          expected_dst.feed_from(dst);
        }
      } else {
        // The format of given dst buffer is expected
        expected_dst = dst;
      }
      args.insert({DNNL_ARG_DST, expected_dst});
      primitive.execute(stream::default_stream(), args);
      // reorder back to dst's buffer if needed
      if (dst.is_empty() || dst.get_desc() == expected_dst.get_desc() ||
          !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
        dst = expected_dst;
      } else {
        dst.feed_from(expected_dst);
      }
    } else {
      args.insert({DNNL_ARG_DST, dst});
      primitive.execute(stream::default_stream(), args);
    }
  }

  // For dynamic int8 op (fp32 * int8 -> fp32)
  // Set reorder flags to false if you are sure the memory layout aligns
  // with primitive descriptor. Otherwise, checks are made and reorder
  // may be needed.
  template <bool with_bias, bool reorder_weight = true>
  static inline void do_compute_dynamic_quant(
      const matmul_forward_params& param, const tensor& src,
      const tensor& weights, const tensor& bias, tensor& dst,
      const scale_t& src_scales, const zero_point_t& src_zero_points,
      const float dst_coeff = 1.0f,
      const engine& aengine = engine::cpu_engine()) {
    /* Compute for dynamic quantized linear. This function does the following
     * things:
     * - Get matmul primitive from param
     * - Get reorder primitive for src from param.dq_param_ptr
     * - Prepare tensors of output scales and zero points.
     * - Compute by executing matmul primitive
     */

    // Get primitive, etc. from param
    IDEEP_ENFORCE(param.dq_param_ptr,
                  "Parameters for dynamic quantization not found");
    auto& pd = param.pd;
    auto& primitive = param.primitive;
    auto& weights_attr = param.weights_attr;
    auto& weights_scales_in = param.dq_param_ptr->weight_scales;
    auto& expected_src_desc = param.dq_param_ptr->src_desc;
    auto& wei_zero_point_m = param.dq_param_ptr->wei_zero_point_m;
    auto& src_reorder = param.dq_param_ptr->src_reorder;

    // Prepare tensor of output scales
    int scale_size = (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
    auto src_scales_in =
        src.has_scale() ? src.get_scale()
                        : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);
    auto& dst_scales_in = IDEEP_DEF_SCALE;

    const dim scale_zp_stride = 1;
    tensor::desc scales_desc = {
        {scale_size}, data_type::f32, {scale_zp_stride}};
    tensor scales_m(scales_desc, aengine);
    auto s = reinterpret_cast<float*>(scales_m.get_data_handle());
    for (memory::dim i = 0; i < scale_size; ++i) {
      s[i] = dst_coeff * dst_scales_in[0] /
             (src_scales_in[0] * weights_scales_in[i]);
    }

    // Prepare tensor of src scales
    int src_scale_size = src_scales_in.size();
    tensor::desc src_scales_desc = {
        {src_scale_size}, data_type::f32, {scale_zp_stride}};
    tensor src_scales_m(src_scales_desc,
                        reinterpret_cast<float*>(src_scales_in.data()),
                        aengine);

    // Prepare tensor of src zero point
    auto src_zero_point =
        src.has_zero_point()
            ? src.get_zero_point()
            : src_zero_points.empty() ? IDEEP_DEF_ZP : src_zero_points;
    const auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
    IDEEP_ENFORCE(src_zero_point_size == 1,
                  "DNNL only support 1-dim zero_point");
    tensor::desc src_zero_point_desc = {
        {src_zero_point_size}, data_type::s32, {scale_zp_stride}};
    tensor src_zero_point_m(src_zero_point_desc,
                            reinterpret_cast<int32_t*>(src_zero_point.data()),
                            aengine);

    // Reroder src (f32 -> u8)
    tensor expected_src(expected_src_desc);
    src_reorder.execute(
        stream::default_stream(),
        {{DNNL_ARG_FROM, src},
         {DNNL_ARG_TO, expected_src},
         {DNNL_ARG_ATTR_OUTPUT_SCALES, src_scales_m},
         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, src_zero_point_m}});

    // Check weight desc
    auto& expected_weights =
        reorder_weight
            ? weights.reorder_if_differ_in(pd.weights_desc(), weights_attr)
            : weights;
    tensor& expected_dst = dst;

    tensor scratchpad(pd.scratchpad_desc());
    if (with_bias) {
      primitive.execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_WEIGHTS, expected_weights},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, bias},
           {DNNL_ARG_DST, expected_dst},
           {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
    } else {
      primitive.execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_WEIGHTS, expected_weights},
           {DNNL_ARG_DST, expected_dst},
           {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
           {DNNL_ARG_SCRATCHPAD, scratchpad}});
    }
  }

  template <bool with_bias>
  static void compute_impl(const tensor& src, const tensor& weights,
                           const tensor& bias, tensor& dst,
                           const float dst_coeff = 1.0f,
                           const float sum_coeff = 1.0f,
                           const scale_t& src_scales = scale_t(),
                           const scale_t& weights_scales = scale_t(),
                           const scale_t& dst_scales = scale_t(),
                           const attr_t& attr = attr_t(),
                           const data_type dst_type = data_type::undef,
                           const lowp_kind alowp_kind = u8s8,
                           const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(src.ndims() == weights.ndims(),
                  "Invalid dims in src or weights");

    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    scale_t dst_scales_in;
    auto dst_data_type = data_type::f32;

    tensor::dims dst_dims = {src.get_dim(0), weights.get_dim(1)};
    auto ndims = weights.ndims();
    if (ndims == 3)
      dst_dims = {src.get_dim(0), src.get_dim(1), weights.get_dim(2)};

    auto weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;
    tensor scales_m, src_zero_point_m, wei_zero_point_m, dst_zero_point_m;
    if (!weights_scales_in.empty()) {
      // weight must be s8 according to
      // https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html
      IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported low-precision combination");

      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);
      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8, tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      int weight_scale_size =
          (weights_scales_in.size() > 1) ? weights.get_dim(1) : 1;
      weights_desc = weights.get_desc();
      if (weights.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(weight_scale_size, false),
                        weights_scales_in};
      }

      // determine dst data type
      if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else {
        dst_data_type = dst_type;
      }

      // fill primitive attr
      scale_t op_scales(weight_scale_size), bias_scales(weight_scale_size);
      dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32)
                          ? IDEEP_DEF_SCALE
                          : dst_scales;
      auto src_zero_point =
          src.has_zero_point() ? src.get_zero_point() : std::vector<int32_t>(1);
      auto src_zero_point_size = static_cast<dim>(src_zero_point.size());
      auto dst_zero_point =
          dst.has_zero_point() ? dst.get_zero_point() : std::vector<int32_t>(1);
      auto dst_zero_point_size = static_cast<dim>(dst_zero_point.size());
      IDEEP_ENFORCE(src_zero_point_size == 1 && dst_zero_point_size == 1,
                    "DNNL only support 1-dim zero_point");
      // according to here:
      // https://oneapi-src.github.io/oneDNN/struct_dnnl_primitive_attr-2.html#doxid-structdnnl-1-1primitive-attr-1aee82deb014cf9702ceb3e725156c25a1
      // zero points can be a vector of the same length as weight_scale_size
      auto wei_zero_point = weights.has_zero_point()
                                ? weights.get_zero_point()
                                : std::vector<int32_t>(weight_scale_size);
      dim wei_zero_point_size = weight_scale_size;

      if (attr.has_op_kind(kind::sum)) {
        float sum_scale = sum_coeff * dst_scales_in[0] /
                          (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        op_attr = attr_t::fuse_sum(sum_scale);
      }

      auto bias_scales_in =
          bias.has_scale() ? bias.get_scale() : IDEEP_DEF_SCALE;
      bias_scales_in =
          bias_scales_in.size() == 1
              ? std::vector<float>(weight_scale_size, bias_scales_in[0])
              : bias_scales_in;
      bool flag_runtime = false;
      if (flag_runtime) {
        op_attr.set_output_scales(utils::op_scale_mask(weight_scale_size),
                                  {DNNL_RUNTIME_F32_VAL});
        tensor::desc scales_desc = {{weight_scale_size}, data_type::f32, {1}};
        scales_m.init(scales_desc, aengine);
        auto s = reinterpret_cast<float*>(scales_m.get_data_handle());
        for (memory::dim i = 0; i < weight_scale_size; ++i) {
          bias_scales[i] = src_scales_in[0] * weights_scales_in[i] /
                           (dst_coeff * bias_scales_in[i]);
          s[i] = dst_coeff * dst_scales_in[0] /
                 (src_scales_in[0] * weights_scales_in[i]);
        }

        op_attr.set_zero_points(DNNL_ARG_SRC, utils::tensor_zp_mask(1),
                                {DNNL_RUNTIME_S32_VAL});
        tensor::desc src_zero_point_desc = {
            {src_zero_point_size}, data_type::s32, {1}};
        src_zero_point_m.init(src_zero_point_desc, aengine);
        auto src_z =
            reinterpret_cast<int32_t*>(src_zero_point_m.get_data_handle());
        for (memory::dim i = 0; i < src_zero_point_size; ++i)
          src_z[i] = src_zero_point[i];

        op_attr.set_zero_points(DNNL_ARG_WEIGHTS, utils::tensor_zp_mask(1),
                                {DNNL_RUNTIME_S32_VAL});
        tensor::desc wei_zero_point_desc = {
            {wei_zero_point_size}, data_type::s32, {1}};
        wei_zero_point_m.init(wei_zero_point_desc, aengine);
        auto wei_z =
            reinterpret_cast<int32_t*>(wei_zero_point_m.get_data_handle());
        for (memory::dim i = 0; i < wei_zero_point_size; ++i)
          wei_z[i] = wei_zero_point[i];

        if (dst_data_type != data_type::f32) {
          op_attr.set_zero_points(DNNL_ARG_DST, utils::tensor_zp_mask(1),
                                  {DNNL_RUNTIME_S32_VAL});
          tensor::desc dst_zero_point_desc = {
              {dst_zero_point_size}, data_type::s32, {1}};
          dst_zero_point_m.init(dst_zero_point_desc, aengine);
          auto dst_z =
              reinterpret_cast<int32_t*>(dst_zero_point_m.get_data_handle());
          for (memory::dim i = 0; i < dst_zero_point_size; ++i)
            dst_z[i] = dst_zero_point[i];
        }
      } else {
        for (int i = 0; i < weight_scale_size; i++) {
          bias_scales[i] = (dst_coeff * bias_scales_in[i]) /
                           (src_scales_in[0] * weights_scales_in[i]);
          // according to here:
          // https://github.com/oneapi-src/oneDNN/blob/51ad89de16e35f5212ad96511bf3074808830894/doc/programming_model/attributes_quantization.md
          // output scales should be src_scales * weight_scales / dst_scales
          op_scales[i] = dst_coeff * src_scales_in[0] * weights_scales_in[i] /
                         dst_scales_in[0];
        }
        op_attr.set_output_scales(utils::op_scale_mask(weight_scale_size),
                                  op_scales);
        op_attr.set_zero_points(DNNL_ARG_SRC,
                                utils::tensor_zp_mask(src_zero_point.size()),
                                src_zero_point);
        op_attr.set_zero_points(DNNL_ARG_WEIGHTS, utils::tensor_zp_mask(1),
                                std::vector<int32_t>(1, wei_zero_point[0]));
        if (dst_data_type != data_type::f32) {
          op_attr.set_zero_points(DNNL_ARG_DST,
                                  utils::tensor_zp_mask(dst_zero_point.size()),
                                  dst_zero_point);
        }
      }

      if (with_bias) {
        tag bia_tag = bias.get_dims().size() == 2 ? tag::ab : tag::abc;
        bias_desc = {bias.get_dims(), data_type::s32, bia_tag};
        if (bias.get_data_type() != data_type::s32) {
          auto ndims = bias.get_dims().size();
          int mask = weight_scale_size > 1 ? 1 << (ndims - 1) : 0;
          bias_attr = {mask, bias_scales};
        }
      }
    } else {
      op_attr = attr;
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      // We intentionally didn't set weight desc to format `any` so DNNL
      // wouldn't have to determine weight format for us. Because the weight
      // tensor from pytorch may have a transposed format (say `ba`). However,
      // DNNL would choose plain format for it by default (`ab` in this case),
      // which would introduces *an extra reorder* afterwards. Here we keep the
      // weight format untouched thanks to optimizations for both plain and
      // transposed formats in DNNL.
      IDEEP_ENFORCE(weights.get_data_type() == data_type::f32 ||
                        weights.get_data_type() == data_type::bf16,
                    "Incorrect data type in weights");
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = src.get_desc().to_type(dst_data_type);
      weights_desc = weights.get_desc().to_type(dst_data_type);
      if (with_bias) {
        IDEEP_ENFORCE(bias.get_data_type() == data_type::f32 ||
                          bias.get_data_type() == data_type::bf16,
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc().to_format_any();
        auto bias_scales = scale_t(1, 1.0 / dst_coeff);
        bias_attr = {utils::tensor_scale_mask(1, false), bias_scales};
      }

      if (attr.has_op_kind(kind::sum)) {
        op_attr = attr_t::fuse_sum(sum_coeff);
      }
      int scale_size = 1;
      op_attr.set_output_scales(utils::op_scale_mask(scale_size),
                                std::vector<float>(1, dst_coeff));
    }

    dst_data_type = dst_type == data_type::undef ? dst_data_type : dst_type;
    tensor::desc dst_desc(dst_dims, dst_data_type, tag::any);
    auto pd = with_bias ? primitive_desc(
                              {src_desc, weights_desc, bias_desc, dst_desc},
                              op_attr, aengine)
                        : primitive_desc({src_desc, weights_desc, dst_desc},
                                         op_attr, aengine);
    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights =
        weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_possible(pd.dst_desc());
    if (!dst_scales.empty() && dst_data_type != data_type::f32) {
      dst.set_scale(dst_scales_in);
    }
    if (with_bias) {
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      super(pd).execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_WEIGHTS, expected_weights},
           {DNNL_ARG_BIAS, expected_bias},
           {DNNL_ARG_DST, dst},
           {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
    } else {
      super(pd).execute(
          stream::default_stream(),
          {{DNNL_ARG_SRC, expected_src},
           {DNNL_ARG_WEIGHTS, expected_weights},
           {DNNL_ARG_DST, dst},
           {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
           {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
    }
  }
};

}  // namespace ideep

#endif

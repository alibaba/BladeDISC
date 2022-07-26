from torch_blade.config import Config

try:
    import torch_blade._torch_blade._quantization as _quantization
    _is_available = True

except ImportError as e:
    _is_available = False


def _jit_pass_quantization_preprocess(c_module):
    if not _is_available:
        return

    cfg = Config.get_current_context_or_new()
    is_enabled_quantization = cfg.enable_int8
    if is_enabled_quantization:
        # Add placeholder for each fake quant of weight.
        # Or it will be folded by _jit_pass_constant_propagation.
        # TODO: remove this when fake_quant is added to the skip_list
        # of _jit_pass_constant_propagation.
        # https://github.com/pytorch/pytorch/issues/81460
        _quantization.add_placeholder_for_fake_quant(c_module)


def _jit_pass_quantization_postprocess(c_module):
    if not _is_available:
        return

    cfg = Config.get_current_context_or_new()
    is_enabled_quantization = cfg.enable_int8
    if _is_available and is_enabled_quantization:
        _quantization.remove_placeholder(c_module)

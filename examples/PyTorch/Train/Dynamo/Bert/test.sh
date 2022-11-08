export TORCH_MHLO_OP_WHITE_LIST="aten::clone;aten::var;aten::rsub;aten::amax;aten::to;aten::tanh;aten::_to_copy"
python3 test_bert.py --backend aot_disc  2>&1 | tee disc.log

import argparse
import os
import time
import types
import torch
import numpy as np
import torch_blade
from torch_blade import Config

os.environ["DISC_ENABLE_STITCH"] = "true"
os.environ["DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL"] = "true"

# Tools for profiling, to be removed in the final release.
_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)

class BalanceBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, N=4):
        mask_score = inputs.clone()
        mask_score = mask_score.reshape(-1, mask_score.shape[1]//N, N)
        prune_indices = torch.argsort(mask_score.reshape(-1, N), 1)
        prune_indices = prune_indices[:, :N//2] + N*torch.range(0, mask_score.numel()//N-1).int().unsqueeze(1).to(mask_score.device)
        mask = torch.ones(mask_score.shape).type_as(inputs).detach()
        mask.reshape(-1)[prune_indices.reshape(-1)] = 0.0
        return mask.reshape(inputs.shape)

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

def dense_to_sparse(weight):
    mask = BalanceBinarizer.apply(weight.abs(), 4)
    return (weight * mask)


parser = argparse.ArgumentParser(description='Inference Benchmark')
parser.add_argument('--model_name', default='bert', help='Path to Engine Model')
parser.add_argument('--batch_size', default=1, help='batch size', type=int)
parser.add_argument('--seq_len', default=128, help='batch size', type=int)
parser.add_argument('--sparse', default=True, help='sparse', type=bool)
args, _ = parser.parse_known_args()

model_name = args.model_name
max_batch_size = args.batch_size
seq_len = args.seq_len

if model_name == 'bert':
    from transformers import BertModel, BertTokenizer, BertConfig, TensorType
    # from bert_model import BertModel
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024, torchscript=True)
    config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)
    # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, torchscript=True)
    model = BertModel(config)
    model.eval()
    model = model.cuda()

    model = model.half()

    dummy_input = (torch.ones(max_batch_size, seq_len).long().cuda(), torch.ones(max_batch_size, seq_len).long().cuda())

if args.sparse:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'pooler' not in name:
            module.weight.data = dense_to_sparse(module.weight.data)
            print(name)

model = torch.jit.trace(model, dummy_input)

with torch.no_grad():

    blade_model = torch_blade.optimize(model, allow_tracing=True, model_inputs=dummy_input)

    org_output = model(*dummy_input)
    blade_output = blade_model(*dummy_input)

    print(torch.mean(1.0*(org_output[0] - blade_output[0] < 1e-2)))
    print(torch.mean(1.0*(org_output[1] - blade_output[1] < 1e-2)))

    torch.jit.save(blade_model, 'bert_base_on.disc.pt')
    # torch.jit.save(blade_model, 'bert_base_off.disc.pt')

    for i in range(100):
        _ = model(*dummy_input)
    
    all_time = 0
    for i in range(1000):
        # torch.cuda.synchronize()
        begin_time = time.time()
        _ = model(*dummy_input)
        # torch.cuda.synchronize()
        end_time = time.time()
        all_time += end_time - begin_time
    
    print("dense:", all_time)

    blade_model = torch.jit.load('bert_base_on.disc.pt').cuda().eval()
    # blade_model = torch.jit.load('bert_base_off.disc.pt').cuda().eval()

    for i in range(100):
        _ = blade_model(*dummy_input)
    
    all_time = 0
    for i in range(1000):
        # torch.cuda.synchronize()
        begin_time = time.time()
        _ = blade_model(*dummy_input)
        # torch.cuda.synchronize()
        end_time = time.time()
        all_time += end_time - begin_time
    
    print("sparse:", all_time)

    # profile start
    cu_prof_start()
    _ = blade_model(*dummy_input)
    cu_prof_stop()
    # profile end
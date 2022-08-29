# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is copied from unifold and added foldacc optimization.
import argparse
import gzip
import logging
import numpy as np
import os

import time
import torch
import json
import pickle
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.data import residue_constants, protein
from unifold.dataset import load_and_process, UnifoldDataset
from unicore.utils import (
    tensor_tree_map,
)

from foldacc.fold.unifold import optimize_unifold


def automatic_chunk_size(seq_len):
    if seq_len < 512:
        chunk_size = 256
    elif seq_len < 1024:
        chunk_size = 128
    elif seq_len < 2048:
        chunk_size = 32
    elif seq_len < 3072:
        chunk_size = 16
    else:
        chunk_size = 1
    return chunk_size


def load_feature_for_one_target(
    config, data_folder, seed=0, is_multimer=False, use_uniprot=False
):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ["A"]
        if use_uniprot:
            uniprot_msa_dir = data_folder

    else:
        uniprot_msa_dir = data_folder
        sequence_ids = open(os.path.join(data_folder, "chains.txt")).readline().split()
    batch, _ = load_and_process(
        config=config.data,
        mode="predict",
        seed=seed,
        batch_idx=None,
        data_idx=0,
        is_distillation=False,
        sequence_ids=sequence_ids,
        monomer_feature_dir=data_folder,
        uniprot_msa_dir=uniprot_msa_dir,
    )
    batch = UnifoldDataset.collater([batch])
    return batch


def main(args):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    args.model_device = local_rank

    config = model_config(args.model_name)
    config.data.common.max_recycling_iters = args.max_recycling_iters
    config.globals.max_recycling_iters = args.max_recycling_iters
    config.data.predict.num_ensembles = args.num_ensembles
    is_multimer = config.model.is_multimer
    if args.sample_templates:
        # enable template samples for diversity
        config.data.predict.subsample_templates = True
    # faster prediction with large chunk
    config.globals.chunk_size = 128
    model = AlphaFold(config)

    print("start to load params {}".format(args.param_path))
    state_dict = torch.load(args.param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(args.model_device)
    model.eval()
    if args.bf16:
        model.bfloat16()
    
    with torch.no_grad():
        model = optimize_unifold(
            model, config, 
            enable_disc=True, 
            enable_low_mem=False,
            dtype=torch.half, 
            device=local_rank, 
            save_dir="tmp")

    # data path is based on target_name
    data_dir = os.path.join(args.data_dir, args.target_name)
    output_dir = os.path.join(args.output_dir, args.target_name)
    os.system("mkdir -p {}".format(output_dir))
    cur_param_path_postfix = os.path.split(args.param_path)[-1]
    name_postfix = ""
    if args.sample_templates:
        name_postfix += "_st"
    if not is_multimer and args.use_uniprot:
        name_postfix += "_uni"
    if args.max_recycling_iters != 3:
        name_postfix += "_r" + str(args.max_recycling_iters)
    if args.num_ensembles != 2:
        name_postfix += "_e" + str(args.num_ensembles)

    print("start to predict {}".format(args.target_name))
    plddts = {}
    ptms = {}
    for seed in range(args.times):
        batch = None
        if torch.distributed.get_rank() == 0:
            cur_seed = hash((args.data_random_seed, seed)) % 100000
            batch = load_feature_for_one_target(
                config,
                data_dir,
                cur_seed,
                is_multimer=is_multimer,
                use_uniprot=args.use_uniprot,
            )
        
        batch = [batch]
            
        torch.distributed.broadcast_object_list(batch, src=0)
        batch = batch[0]

        seq_len = batch["aatype"].shape[-1]
        model.globals.chunk_size = automatic_chunk_size(seq_len)

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=args.model_device)
                for k, v in batch.items()
            }

            shapes = {k: v.shape for k, v in batch.items()}
            print(shapes)
            torch.cuda.synchronize()
            t = time.perf_counter()
            raw_out = model(batch)
            torch.cuda.synchronize()
            print(f"Inference time: {time.perf_counter() - t}")
        
        torch.distributed.barrier()

        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x

        if torch.distributed.get_rank() == 0:
            if not args.save_raw_output:
                score = ["plddt", "ptm", "iptm", "iptm+ptm"]
                out = {
                        k: v for k, v in raw_out.items()
                        if k.startswith("final_") or k in score
                    }
            else:
                out = raw_out
            del raw_out
            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
            batch = tensor_tree_map(to_float, batch)
            out = tensor_tree_map(lambda t: t[0, ...], out)
            out = tensor_tree_map(to_float, out)
            batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            plddt = out["plddt"]
            mean_plddt = np.mean(plddt)
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1
            )
            # TODO: , may need to reorder chains, based on entity_ids
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )
            cur_save_name = (
                f"{args.model_name}_{cur_param_path_postfix}_{cur_seed}{name_postfix}"
            )
            plddts[cur_save_name] = str(mean_plddt)
            if is_multimer:
                ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))
            with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
                f.write(protein.to_pdb(cur_protein))
            if args.save_raw_output:
                with gzip.open(os.path.join(output_dir, cur_save_name + '_outputs.pkl.gz'), 'wb') as f:
                    pickle.dump(out, f)
            del out
            
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        print("plddts", plddts)
        score_name = f"{args.model_name}_{cur_param_path_postfix}_{args.data_random_seed}_{args.times}{name_postfix}"
        plddt_fname = score_name + "_plddt.json"
        json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
        if ptms:
            print("ptms", ptms)
            ptm_fname = score_name + "_ptm.json"
            json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)

    torch.distributed.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda:0",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_2",
    )
    parser.add_argument(
        "--param_path", type=str, default=None, help="Path to model parameters."
    )
    parser.add_argument(
        "--data_random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_recycling_iters",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=2,
    )
    parser.add_argument("--sample_templates", action="store_true")
    parser.add_argument("--use_uniprot", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_raw_output", action="store_true")

    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying
            --model_device for better performance"""
        )

    main(args)

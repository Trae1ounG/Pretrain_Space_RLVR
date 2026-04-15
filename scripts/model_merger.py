# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
)
from safetensors import safe_open
SAFETENSORS_AVAILABLE=1
def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def load_lora_adapter(adapter_path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load LoRA adapter from safetensors file"""
    adapter_weights = {}
    
    # Load adapter config
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: adapter_config.json not found at {config_path}")
        return {}, {}
    
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    print(f"Loading LoRA adapter from {adapter_path}")
    print(f"Adapter config: {adapter_config}")
    
    # Check if adapter file exists
    if not os.path.exists(adapter_path):
        print(f"Error: LoRA adapter file not found: {adapter_path}")
        return {}, {}
    
    # Try different methods to load the safetensors file
    try:
        if SAFETENSORS_AVAILABLE:
            # Method 1: Use load_file (simpler approach)
            try:
                adapter_weights = load_file(adapter_path, device="cpu")
                print(f"Loaded {len(adapter_weights)} LoRA adapter parameters using load_file")
            except Exception as e1:
                print(f"load_file method failed: {e1}")
                # Method 2: Use safe_open
                try:
                    # with open(adapter_path, 'rb') as f:
                    #     file_content = f.read()
                    # adapter_weights = safetensors.torch.load(file_content, device="cpu")
                    with safe_open(adapter_path+"/adapter_model.safetensors", framework="pt") as f:
                        for key in f.keys():
                            adapter_weights[key] = f.get_tensor(key)
                    print(f"Loaded {len(adapter_weights)} LoRA adapter parameters using safe_open")
                except Exception as e2:
                    print(f"safe_open method also failed: {e2}")
                    return {}, {}
        else:
            print("Error: safetensors library not available")
            return {}, {}
            
    except Exception as e:
        print(f"Error loading adapter weights: {e}")
        return {}, {}
    
    return adapter_weights, adapter_config


def merge_lora_weights(base_state_dict: Dict[str, torch.Tensor], 
                      lora_weights: Dict[str, torch.Tensor],
                      adapter_config: Dict) -> Dict[str, torch.Tensor]:
    """Merge LoRA weights into base model weights"""
    
    # Extract LoRA configuration
    r = adapter_config.get("r", 8)
    lora_alpha = adapter_config.get("lora_alpha", 16)
    scaling = lora_alpha / r
    
    print(f"LoRA config - r: {r}, lora_alpha: {lora_alpha}, scaling: {scaling}")
    
    # Group LoRA weights by module
    lora_a_weights = {}
    lora_b_weights = {}
    
    for key, weight in lora_weights.items():
        if ".lora_A." in key:
            module_name = key.replace(".lora_A.weight", "")
            lora_a_weights[module_name] = weight
        elif ".lora_B." in key:
            module_name = key.replace(".lora_B.weight", "")
            lora_b_weights[module_name] = weight
    
    print(f"Found LoRA A weights for {len(lora_a_weights)} modules")
    print(f"Found LoRA B weights for {len(lora_b_weights)} modules")
    
    # Merge LoRA weights into base weights
    merged_state_dict = base_state_dict.copy()
    
    for module_name in lora_a_weights:
        if module_name in lora_b_weights:
            # Calculate delta weight: scaling * B @ A
            lora_a = lora_a_weights[module_name]
            lora_b = lora_b_weights[module_name]
            
            delta_weight = scaling * (lora_b @ lora_a)
            
            # Find corresponding base weight
            base_weight_key = None
            for key in base_state_dict:
                if module_name in key and key.endswith(".weight") and "lora" not in key:
                    base_weight_key = key
                    break
            if base_weight_key and base_weight_key in merged_state_dict:
                print(f"Merging LoRA for {module_name} -> {base_weight_key}")
                rm_key = base_weight_key.replace("base_layer.","")
                rm_key = rm_key.replace("base_model.model.","")
                # import pdb; pdb.set_trace()
                merged_state_dict[rm_key] = merged_state_dict[base_weight_key] + delta_weight
            else:
                print(f"Warning: Could not find base weight for {module_name}")
    new_state_dict = {}
    for k,v in merged_state_dict.items():
        if "post_attention_layernorm"  in k  or "input_layernorm"  in k or "embed_tokens" in k or "norm" in k:
            k = k.replace("base_model.model.","")  
        if  "bias" in k:
            k = k.replace("base_model.model.","")  
            k = k.replace("base_layer.","")  
        if "base_model" in k and not "lora" in k and "base_layer" not in k:
            k = k.replace("base_model.model.","")  
            if k not in new_state_dict.keys():
                new_state_dict[k] = v
                continue
        if "base_layer" in k or "lora" in k or "base_model" in k: 
            # import pdb; pdb.set_trace()
            continue
        new_state_dict[k] = v
    # import pdb; pdb.set_trace()
    return new_state_dict


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    parser.add_argument("--lora_adapter_path", default=None, type=str, help="Path to LoRA adapter model.safetensors file")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: Dict[str, List[torch.Tensor]] = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    args.lora_adapter_path = args.local_dir + "/lora_adapter"
    # Load and merge LoRA adapter if provided
    if args.lora_adapter_path and os.path.exists(args.lora_adapter_path):
        print(f"Loading LoRA adapter from {args.lora_adapter_path}")
        lora_weights, adapter_config = load_lora_adapter(args.lora_adapter_path)
        if lora_weights:
            print("Merging LoRA weights into base model...")
            state_dict = merge_lora_weights(state_dict, lora_weights, adapter_config)
            print("LoRA weights merged successfully.")
        else:
            print("Warning: No LoRA weights loaded, skipping merge.")
    elif args.lora_adapter_path:
        print(f"Warning: LoRA adapter path provided but file not found: {args.lora_adapter_path}")

    hf_path = os.path.join(local_dir, "huggingface")
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: List[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model

    # if args.hf_upload_path:
    #     upload_model_to_huggingface(hf_path, args.hf_upload_path)
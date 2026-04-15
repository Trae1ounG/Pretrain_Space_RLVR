# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
import torch.nn.functional as F
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_, fsdp2_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from collections import defaultdict
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits
            
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    # Add input embed entropy
    def _forward_micro_batch_entropy(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        response_length = micro_batch["responses"].size(-1)
        
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                def calulate_entropy_(logits):
                    embed_logits_rmpad = logits.squeeze(0)  # (total_nnz, vocab_size)
                    embed_logits_rmpad.div_(temperature)
                    embed_entropy_rmpad = self.compute_entropy_from_logits(embed_logits_rmpad)
                    embed_full_entropy = pad_input(
                                hidden_states=embed_entropy_rmpad.unsqueeze(-1),
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            ) 
                    embed_entropy = embed_full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                    return embed_entropy
                output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        **extra_args,
                    )  # prevent model thinks we are generating
                entropy_stats = {
                    'input': defaultdict(),
                    'aft_innorm': defaultdict(),
                    'aft_attn': defaultdict(),
                    'aft_attn_add': defaultdict(),
                    'aft_attn_addnorm': defaultdict(),
                    'aft_mlp_add': defaultdict(),
                }
                
                with FSDP.summon_full_params(self.actor_module, recurse=True, writeback=False):
                    with torch.no_grad():
                        base_model = self.actor_module._fsdp_wrapped_module.model
                        model = self.actor_module._fsdp_wrapped_module
                        layers = base_model.layers
                        for layer_idx in range(len(layers)):
                            residual, hidden_states = output.hidden_states[layer_idx], output.hidden_states[layer_idx]
                            logits = model.lm_head(hidden_states)
                            entropy_stats['input'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            hidden_states = layers[layer_idx].input_layernorm(hidden_states)
                            logits = model.lm_head(hidden_states)
                            entropy_stats['aft_innorm'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            position_embeddings = base_model.rotary_emb(hidden_states, position_ids_rmpad)
                            
                            hidden_states = layers[layer_idx].self_attn(hidden_states,attention_mask=None, position_ids=position_ids_rmpad,position_embeddings=position_embeddings,**extra_args,)[0]
                            logits = model.lm_head(hidden_states)
                            entropy_stats['aft_attn'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            hidden_states += residual 
                            logits = model.lm_head(hidden_states)
                            entropy_stats['aft_attn_add'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            residual = hidden_states
                            hidden_states = layers[layer_idx].post_attention_layernorm(hidden_states)
                            logits = model.lm_head(hidden_states)
                            entropy_stats['aft_attn_addnorm'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            
                            hidden_states = residual + layers[layer_idx].mlp(hidden_states)
                            logits = model.lm_head(hidden_states)
                            entropy_stats['aft_mlp_add'][layer_idx] = calulate_entropy_(logits).detach().cpu()
                            del logits
                            del hidden_states, residual
                            torch.cuda.empty_cache()
                        # breakpoint()
                topk_probs_rmpad = None
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    # embed_logits_rmpad = logits.squeeze(0)  # (total_nnz, vocab_size)
                    # embed_logits_rmpad.div_(temperature)
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )


                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
              

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    # embed_full_entropy = pad_input(
                    #     hidden_states=embed_entropy_rmpad.unsqueeze(-1),
                    #     indices=indices,
                    #     batch=batch_size,
                    #     seqlen=seqlen,
                    # )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_probs = None
               

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
            # breakpoint()
            return entropy, entropy_stats , log_probs
    
    def _compute_topk_probs(self, logits: torch.Tensor, k: int = 3) -> torch.Tensor:
        logits = logits.float()
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_logits, _ = torch.topk(logits, k=k, dim=-1)
        topk_probs = torch.exp(topk_logits - lse)           # (..., k)
        eos_prob = torch.exp(logits[..., self.tokenizer.eos_token_id] - lse.squeeze(-1))
        eos_prob = eos_prob.unsqueeze(-1)                            # (..., 1)
        topk_with_eos = torch.cat([topk_probs, eos_prob], dim=-1) 
        return topk_with_eos

    def _forward_micro_batch_layer_k(
        self, micro_batch, temperature, calculate_entropy=False, layer_k=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        response_length = micro_batch["responses"].size(-1)
        
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding: # This way
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        **extra_args,
                    )  # prevent model thinks we are generating
                if self.use_fused_kernels: # Do not use
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else: # This way
                    # Logits from inner layer
                    logits_rmpad = output.mid_layer_logits[layer_k].squeeze(0)
                    logits_rmpad.div_(temperature)
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    del output.mid_layer_logits,logits_rmpad, output.hidden_states
                    torch.cuda.empty_cache()
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False

                    logits_rmpad = output.logits.squeeze(0)
                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    del logits_rmpad, output
                    torch.cuda.empty_cache()
                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
              

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy , log_probs
    
   
    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_topk_probs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            self_certainty = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # rmpad branch
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    **extra_args,
                )  # prevent model thinks we are generating

                topk_probs = None
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if return_topk_probs:
                        topk_probs = self._compute_topk_probs(logits_rmpad, k=100)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_topk_probs and topk_probs is not None:
                        topk_probs = gather_outputs_and_unpad(
                            topk_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_probs = None
                if return_topk_probs and topk_probs is not None:
                    full_topk_probs = pad_input(
                        hidden_states=topk_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_topk_probs and full_topk_probs is not None:
                    topk_probs = full_topk_probs[:, -response_length - 1 : -1, :]

                if return_topk_probs:
                    return entropy, log_probs, topk_probs
                return entropy, log_probs

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    topk_probs = self._compute_topk_probs(logits, k=3) if return_topk_probs else None
            return entropy, log_probs

    def _forward_micro_batch_topk(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_topk_probs: bool = False,
        topk_indices = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )
        return_topk_indices = return_topk_probs and topk_indices is None
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            self_certainty = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                # rmpad branch
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    **extra_args,
                )  # prevent model thinks we are generating

                topk_probs = None
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    distill_topk = 100
                    if return_topk_probs:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = ulysses_pad_and_slice_inputs(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                        # topk_probs = self._compute_topk_probs(logits_rmpad, k=100)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_topk_probs and topk_probs is not None:
                        # topk_probs = gather_outputs_and_unpad(
                        #     topk_probs,
                        #     gather_dim=0,
                        #     unpad_dim=0,
                        #     padding_size=pad_size,
                        # )
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_topk_probs = None
                if return_topk_probs:
                    # full_topk_probs = pad_input(
                    #     hidden_states=topk_probs,
                    #     indices=indices,
                    #     batch=batch_size,
                    #     seqlen=seqlen,
                    # )
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_topk_probs:
                    # topk_probs = full_topk_probs[:, -response_length - 1 : -1, :]
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

                if return_topk_probs:
                    return entropy, log_probs, topk_logps, topk_indices
                return entropy, log_probs

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    topk_probs = self._compute_topk_probs(logits, k=3) if return_topk_probs else None
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                # Original Process
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss and "ref_log_prob" in data.batch.keys():
            select_keys.append("ref_log_prob")
        if data.meta_info.get("pretrain_loss_coef", 0.0) != 0.0 and self.config.use_kl_loss:
            if "pretrain_ref_log_prob" in data.batch.keys():
                select_keys.append("pretrain_ref_log_prob")

        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")
        
        # Reinforcement Pretraining
        if data.meta_info.get("pretrain_loss_coef", 0.0) != 0.0:
            select_keys.append("pretrain_old_log_probs")
            select_keys.append("eos_mask")
            if "pretrain_advantages" in data.batch.keys():
                select_keys.append("pretrain_advantages")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        print(f"On_policy:{on_policy}")
        def mask_and_mean(x, mask):
            seq_losses = torch.sum(x * mask, dim=-1) / torch.sum(mask, dim=-1)  # token-mean
            loss = torch.mean(seq_losses)  # seq-mean
            return loss
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    # Use separate loss_agg_mode for pretrain if configured
                    pretrain_loss_agg_mode = self.config.get("pretrain_loss_agg_mode", None) or loss_agg_mode

                    # =========================================================================
                    # [Modified Logic] Decoupled Loss Calculation
                    # =========================================================================

                    # 1. RL Loss (Policy Gradient)
                    # We verify if we need positive/negative RL based on config
                    # Default: positive_rl=True (implicit), negative_rl from config
                    
                    # Sample-level mask: determine pos/neg based on per-sample advantage sign
                    # For GRPO, all tokens in a sample share the same advantage value,
                    # so taking any token (e.g. via sum) gives the correct sign.
                    sample_adv_sign = advantages.sum(dim=-1)  # (bsz,)
                    pos_sample_mask = (sample_adv_sign > 0).float().unsqueeze(-1)  # (bsz, 1)
                    neg_sample_mask = (sample_adv_sign < 0).float().unsqueeze(-1)  # (bsz, 1)
                    
                    use_pos_rl = self.config.get("positive_rl", True)
                    use_neg_rl = self.config.get("negative_rl", True)
                    
                    final_rl_advantages = advantages.clone()
                    rl_response_mask = response_mask.clone()
                    
                    if use_pos_rl and not use_neg_rl:
                        final_rl_advantages = final_rl_advantages * pos_sample_mask
                        rl_response_mask = rl_response_mask * pos_sample_mask
                    elif not use_pos_rl and use_neg_rl:
                        final_rl_advantages = final_rl_advantages * neg_sample_mask
                        rl_response_mask = rl_response_mask * neg_sample_mask
                    elif not use_pos_rl and not use_neg_rl:
                        final_rl_advantages = final_rl_advantages * 0.0
                        rl_response_mask = rl_response_mask * 0.0
                    # else: Standard PPO (Pos + Neg) -> Keep original advantages and mask

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    topk_probs = None
                    # BuPO
                    if self.config.internal_policy_interative:
                        if micro_batch.meta_info['global_steps'] <= self.config.iterative_steps:
                            entropy, log_prob = self._forward_micro_batch_layer_k(
                               model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, layer_k=self.config.internal_layer
                            )
                        else:
                            entropy, log_prob, topk_probs = self._forward_micro_batch(
                                model_inputs,
                                temperature=temperature,
                                calculate_entropy=calculate_entropy,
                                return_topk_probs=True,
                            )
                    else:
                        entropy, log_prob, topk_probs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_topk_probs=True,
                        )
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    
                    # Compute MAIN RL Loss with the filtered advantages and masked response
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=final_rl_advantages,
                        response_mask=rl_response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )  

                    # (Removed dead code for explicit negative_pg_loss calculation since we handled it via masking above)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    log_prob_detached = log_prob.detach()
                    micro_batch_metrics["actor/logprob_mean"] = mask_and_mean(
                        log_prob_detached, response_mask
                    ).item()
                    micro_batch_metrics["actor/prob_mean"] = mask_and_mean(
                        torch.exp(log_prob_detached), response_mask
                    ).item()
                    if topk_probs is not None:
                        topk_probs_detached = topk_probs.detach()
                        topk_log_probs_detached = torch.log(topk_probs_detached.clamp_min(1e-12))
                        micro_batch_metrics["actor/top1_prob_mean"] = mask_and_mean(
                            topk_probs_detached[:, :, 0], response_mask
                        ).item()
                        micro_batch_metrics["actor/top2_prob_mean"] = mask_and_mean(
                            topk_probs_detached[:, :, 1], response_mask
                        ).item()
                        micro_batch_metrics["actor/top3_prob_mean"] = mask_and_mean(
                            topk_probs_detached[:, :, 2], response_mask
                        ).item()
                        micro_batch_metrics["actor/eos_prob_mean"] = mask_and_mean(
                            topk_probs_detached[:, :, -1], response_mask
                        ).item()
                        micro_batch_metrics["actor/top1_logprob_mean"] = mask_and_mean(
                            topk_log_probs_detached[:, :, 0], response_mask
                        ).item()
                        micro_batch_metrics["actor/top2_logprob_mean"] = mask_and_mean(
                            topk_log_probs_detached[:, :, 1], response_mask
                        ).item()
                        micro_batch_metrics["actor/top3_logprob_mean"] = mask_and_mean(
                            topk_log_probs_detached[:, :, 2], response_mask
                        ).item()
                        micro_batch_metrics["actor/eos_logprob_mean"] = mask_and_mean(
                            topk_log_probs_detached[:, :, -1], response_mask
                        ).item()

                    # Reinforcement Pretraining
                    pretrain_loss_coef = data.meta_info.get("pretrain_loss_coef", 0.01)
                    pretrain_only = data.meta_info.get("pretrain_only", False)
                    pretrain_max_resp_len = data.meta_info.get("pretrain_max_response_length", None)
                    
                    # Determine whether pretrain is enabled based on parameter combination
                    pretrain_loss_disable_step = data.meta_info.get("pretrain_loss_disable_step", None)
                    pretrain_only_warmup = data.meta_info.get("pretrain_only_warmup", False)
                    current_step = micro_batch.meta_info.get('global_steps', 0)
                    pretrain_enabled = (pretrain_loss_disable_step is not None or pretrain_only == True)
                    
                    if pretrain_loss_disable_step is not None:
                        if current_step >= pretrain_loss_disable_step:
                            pretrain_loss_coef = 0.0
                            pretrain_enabled = False
                            if torch.distributed.get_rank() == 0 and idx == 0 and batch_idx == 0:
                                print(f"INFO: Pretrain loss disabled at step {current_step} (disable_step={pretrain_loss_disable_step})")
                        elif pretrain_only_warmup:
                             pretrain_only = True
                             if torch.distributed.get_rank() == 0 and idx == 0 and batch_idx == 0:
                                print(f"INFO: Pretrain ONLY warmup at step {current_step} (disable_step={pretrain_loss_disable_step})")
                    
                    if pretrain_enabled and "pretrain_old_log_probs" not in model_inputs:
                        if torch.distributed.get_rank() == 0 and idx == 0 and batch_idx == 0:
                            print("WARN: pretrain_enabled but pretrain_old_log_probs not found, disabling pretrain for this batch")
                        pretrain_enabled = False
                        pretrain_only = False
                    
                    pretrain_inputs = {k: v for k, v in model_inputs.items()} # Shallow copy
                    if pretrain_enabled:
                        # Construct pretrain inputs (Response Only)
                        responses = pretrain_inputs["responses"]
                        response_len = pretrain_inputs["responses"].size(-1)
                        seq_len = pretrain_inputs["input_ids"].size(-1)
                        prompt_len = seq_len - response_len
                        
                        # Apply pretrain-specific max response length truncation if configured
                        if pretrain_max_resp_len is not None and response_len > pretrain_max_resp_len:
                            response_len = pretrain_max_resp_len

                        # Slice to keep only response (with truncation if configured)
                        bos_col = torch.full(
                            (responses.size(0), 1),
                            151644,
                            device=responses.device,
                            dtype=responses.dtype,
                        )
                        pretrain_inputs["input_ids"] =  torch.cat([bos_col, pretrain_inputs["input_ids"][:, prompt_len:prompt_len + response_len]], dim=1)
                        attn_col = torch.full(
                            (responses.size(0), 1),
                            1,
                            device=responses.device,
                            dtype=responses.dtype,
                        )
                        pretrain_inputs["attention_mask"] = torch.cat([attn_col, pretrain_inputs["attention_mask"][:, prompt_len:prompt_len + response_len]], dim=1)
                        
                        # Reset position_ids
                        if "position_ids" in pretrain_inputs:
                            device = pretrain_inputs["input_ids"].device
                            bsz = pretrain_inputs["input_ids"].shape[0]
                            # Create position_ids 0..response_len (including BOS)
                            pretrain_inputs["position_ids"] = torch.arange(response_len+1, device=device).unsqueeze(0).expand(bsz, -1)
                        
                    if pretrain_enabled:
                        # Forward pass for p(y) - unconditional probability
                        pretrain_entropy, pretrain_log_prob, pretrain_topk_probs = self._forward_micro_batch(
                            pretrain_inputs,
                            temperature=temperature,
                            calculate_entropy=True,
                            return_topk_probs=True,
                        )
                        pretrain_old = model_inputs["pretrain_old_log_probs"]
                        min_len = min(pretrain_log_prob.size(1), pretrain_old.size(1))
                        # Slice to align shapes
                        p_entropy = pretrain_entropy[:, :min_len]
                        p_log_prob_unconditional = pretrain_log_prob[:, :min_len]  # p(y) - unconditional
                        p_old_log_prob = pretrain_old[:, :min_len]
                        p_response_mask = response_mask[:, :min_len]
                        
                        p_log_prob = p_log_prob_unconditional
                        
                        # Use pretrain_advantages if available (computed with pretrain_norm_adv_by_std), otherwise fallback to advantages
                        if "pretrain_advantages" in model_inputs:
                            pretrain_adv = model_inputs["pretrain_advantages"]
                        else:
                            pretrain_adv = model_inputs["advantages"]
                        p_advantages = pretrain_adv[:, :min_len]
                        pretrain_entropy_loss = agg_loss(loss_mat=pretrain_entropy, loss_mask=p_response_mask, loss_agg_mode=pretrain_loss_agg_mode)
                        pretrain_log_prob_detached = pretrain_log_prob.detach()
                        micro_batch_metrics["actor/pretrain/logprob_mean"] = mask_and_mean(
                            pretrain_log_prob_detached, p_response_mask
                        ).item()
                        micro_batch_metrics["actor/pretrain/prob_mean"] = mask_and_mean(
                            torch.exp(pretrain_log_prob_detached), p_response_mask
                        ).item()
                        if pretrain_topk_probs is not None:
                            pretrain_topk_probs_detached = pretrain_topk_probs.detach()
                            pretrain_topk_log_probs_detached = torch.log(
                                pretrain_topk_probs_detached.clamp_min(1e-12)
                            )
                            micro_batch_metrics["actor/pretrain/top1_prob_mean"] = mask_and_mean(
                                pretrain_topk_probs_detached[:, :, 0], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/top2_prob_mean"] = mask_and_mean(
                                pretrain_topk_probs_detached[:, :, 1], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/top3_prob_mean"] = mask_and_mean(
                                pretrain_topk_probs_detached[:, :, 2], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/eos_prob_mean"] = mask_and_mean(
                                pretrain_topk_probs_detached[:, :, -1], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/top1_logprob_mean"] = mask_and_mean(
                                pretrain_topk_log_probs_detached[:, :, 0], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/top2_logprob_mean"] = mask_and_mean(
                                pretrain_topk_log_probs_detached[:, :, 1], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/top3_logprob_mean"] = mask_and_mean(
                                pretrain_topk_log_probs_detached[:, :, 2], p_response_mask
                            ).item()
                            micro_batch_metrics["actor/pretrain/eos_logprob_mean"] = mask_and_mean(
                                pretrain_topk_log_probs_detached[:, :, -1], p_response_mask
                            ).item()

                        if self.config.negative_pretrain is True and self.config.get("positive_pretrain", True) is False:
                            # Negative Pretrain Only: restrict response_mask to tokens from negative samples
                            # Using response_mask (not adv mask) ensures loss aggregation denominator is correct
                            neg_sample_mask = (p_advantages < 0.0).any(dim=-1, keepdim=True)  # (bsz, 1), sample-level
                            p_response_mask = p_response_mask * neg_sample_mask
                        elif self.config.negative_pretrain is False and self.config.get("positive_pretrain", True) is True:
                            # Positive Pretrain Only: restrict response_mask to tokens from positive samples
                            pos_sample_mask = (p_advantages > 0.0).any(dim=-1, keepdim=True)  # (bsz, 1), sample-level
                            p_response_mask = p_response_mask * pos_sample_mask
                        elif self.config.negative_pretrain is True and self.config.get("positive_pretrain", True) is True:
                            # Both (Pos + Neg) - Standard, no masking needed
                            pass
                        else:
                            # None: zero out response_mask so no tokens contribute to loss
                            p_response_mask = p_response_mask * 0

                        policy_loss_fn = get_policy_loss_fn("vanilla")
                        pretrain_pg_loss, pretrain_pg_clipfrac, pretrain_ppo_kl, _ = policy_loss_fn(
                            old_log_prob=p_old_log_prob,
                            log_prob=p_log_prob,
                            advantages=p_advantages, 
                            response_mask=p_response_mask, 
                            loss_agg_mode=pretrain_loss_agg_mode,
                            config=self.config,
                            rollout_log_probs=None 
                        )
                        # Metrics
                        micro_batch_metrics["actor/pretrain/pg_loss"] = pretrain_pg_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/pretrain/pg_clipfrac"] = pretrain_pg_clipfrac.detach().item()
                        micro_batch_metrics["actor/pretrain/ppo_kl"] = pretrain_ppo_kl.detach().item()
                        micro_batch_metrics["actor/pretrain/entropy"] = pretrain_entropy_loss.detach().item() 

                    # Combine Loss
                    if pretrain_only:
                        policy_loss_final = policy_loss * 0 + pretrain_pg_loss * pretrain_loss_coef
                        eps = 1e-8
                        micro_batch_metrics["actor/pretrain/rl_div_pretrain_ratio"] = policy_loss.detach().item() / (pretrain_pg_loss.detach().item() + eps)
                        micro_batch_metrics["actor/pretrain/pretrain_div_rl_ratio"] = pretrain_pg_loss.detach().item() / (policy_loss.detach().item() + eps)
                    elif not pretrain_enabled:
                        policy_loss_final = policy_loss
                    else:
                        policy_loss_final = policy_loss + (pretrain_pg_loss * pretrain_loss_coef)
                        eps = 1e-8
                        micro_batch_metrics["actor/pretrain/rl_div_pretrain_ratio"] = policy_loss.detach().item() / (pretrain_pg_loss.detach().item() + eps)
                        micro_batch_metrics["actor/pretrain/pretrain_div_rl_ratio"] = pretrain_pg_loss.detach().item() / (policy_loss.detach().item() + eps)

                    # Apply KL Loss
                    use_kl_loss = self.config.use_kl_loss
                    if "ref_log_prob" not in model_inputs:
                        use_kl_loss = False

                    if use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss_final = policy_loss_final + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss_final * loss_scale_factor
                    else:
                        loss = policy_loss_final * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                   
                    append_to_dict(metrics, micro_batch_metrics)
                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

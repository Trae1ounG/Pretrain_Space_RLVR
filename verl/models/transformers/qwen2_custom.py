# 基于现有的qwen2.py修改
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from verl.models.transformers.qwen2 import qwen2_flash_attn_forward

class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 添加中间层logits计算的配置
        self.compute_intermediate_logits = getattr(config, 'compute_intermediate_logits', False)
        self.intermediate_layer_idx = getattr(config, 'intermediate_layer_idx', 26)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        **kwargs
    ):
        # 调用父类的forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # 如果启用中间层logits计算
        if self.compute_intermediate_logits and output_hidden_states and hasattr(outputs, 'hidden_states'):
            with torch.no_grad():  # 避免梯度计算问题
                intermediate_hidden = outputs.hidden_states[self.intermediate_layer_idx]
                intermediate_logits = self.lm_head(intermediate_hidden)
                
                # 将中间层logits添加到输出中
                if hasattr(outputs, 'intermediate_logits'):
                    outputs.intermediate_logits = intermediate_logits
                else:
                    # 创建新的输出对象
                    from dataclasses import dataclass
                    from typing import Optional, Tuple
                    
                    @dataclass
                    class CustomModelOutput:
                        logits: torch.FloatTensor
                        intermediate_logits: Optional[torch.FloatTensor] = None
                        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
                        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
                    
                    outputs = CustomModelOutput(
                        logits=outputs.logits,
                        intermediate_logits=intermediate_logits,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions
                    )
        
        return outputs

# 导出forward函数供monkey patch使用
def forward_with_torch_backend(model, *args, **kwargs):
    return model.forward(*args, **kwargs)

def forward_with_triton_backend(model, *args, **kwargs):
    return model.forward(*args, **kwargs)
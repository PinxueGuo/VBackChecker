#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM)
from transformers.models.llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

import torch.nn.functional as F

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def contrastive_loss(self, hidden_states, labels, seg_token_id, rej_token_id):
        out_hidden_states = hidden_states[:, :-1, :]
        hidd_dim = out_hidden_states.size(-1)
        flat_hidden_states = out_hidden_states.reshape(-1, hidd_dim)
        positive_hidden = flat_hidden_states[labels == seg_token_id]
        negative_hidden = flat_hidden_states[labels == rej_token_id]
        if len(positive_hidden) == 0 or len(negative_hidden) == 0:
            return None

        temperature = 0.07

        # 1. 对正、负样本分别做 L2 归一化
        pos = F.normalize(positive_hidden, dim=-1)  # [a, dim]
        neg = F.normalize(negative_hidden, dim=-1)  # [b, dim]

        # 2. 拼接所有样本
        all_embeddings = torch.cat([pos, neg], dim=0)  # [a+b, dim]
        a, b = pos.size(0), neg.size(0)
        n = a + b  # 总样本数

        # 3. 计算相似度矩阵 (使用点积作为余弦相似度, 因为已经做了 L2 归一化)
        sim = all_embeddings @ all_embeddings.T  # [n, n]

        # 4. 构造同组掩码: 只要下标都在 [0, a) 就是同一组(正样本)，都在 [a, a+b) 就是同一组(负样本)
        device = sim.device
        labels = torch.zeros((n, n), dtype=torch.bool, device=device)
        # positive 样本之间
        labels[:a, :a] = True
        # negative 样本之间
        labels[a:, a:] = True

        # 自己和自己不要算在对比里 (diagonal)，后面会用到
        # 可以用一个 mask 去掉对角线，以避免 log(0)
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)

        # 5. 计算对比学习损失(InfoNCE/NT-Xent)
        #    对第 i 个样本而言:
        #      同组分子: \sum_{j in same(i), j != i} exp(sim(i, j)/temp)
        #      分母: \sum_{k != i} exp(sim(i, k)/temp)
        sim = sim / temperature
        exp_sim = torch.exp(sim) * (~diag_mask)  # 去掉对角线（自己跟自己）

        # 分母: 对每一行求和
        sum_sim = exp_sim.sum(dim=-1)  # [n]

        # 分子: 只保留同组位置
        same_sim = exp_sim * labels
        pos_sim = same_sim.sum(dim=-1)  # [n]

        # 如果某一行同组只有自己，pos_sim 可能为0，这种情况需要做一点平滑(或直接略过)
        # 这里直接 +1e-8 避免出现 log(0)
        loss = -torch.log((pos_sim + 1e-8) / (sum_sim + 1e-8))

        return loss.mean()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            w = torch.ones_like(shift_logits[0])   # 32005
            seg_token_id = 32003
            rej_token_id = 32004
            w[seg_token_id] *= 10.
            w[rej_token_id] *= 10.
            loss = F.cross_entropy(shift_logits, shift_labels, weight=w)
            # cl_loss = self.contrastive_loss(hidden_states, shift_labels, seg_token_id, rej_token_id)
            # if cl_loss is not None:
            #     loss += cl_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel
from torch import nn
import torch
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_flex_attn_available,
    logging,
)
from transformers import GPTNeoXConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import flex_attention

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"


class GPTNeoXForCausalLM_MTP(GPTNeoXForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        # super().__init__(config)

        # self.gpt_neox = GPTNeoXModel(config)
        # self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # # Initialize weights and apply final processing
        # self.post_init()
        self.embed_out_mtp = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.config.eos_token_id = 50256 # We use gpt2 tokenizer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        in_labels = labels

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if in_labels is not None:
            # move labels to correct device to enable model parallelism
            labels = in_labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ntp_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

   
        ntp_logits = lm_logits
        # return CausalLMOutputWithPast(
        #     loss=lm_loss,
        #     logits=lm_logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        hidden_states = outputs[0]
        lm_logits = self.embed_out_mtp(hidden_states)

        lm_loss = None
        if in_labels is not None:
            labels = in_labels.to(lm_logits.device)
            batch_size, seq_len = labels.shape

            # 生成next_eos张量，记录每个位置之后第一个eos的位置
            next_eos = torch.full((batch_size, seq_len), seq_len, dtype=torch.long, device=labels.device)
            eos_mask = labels == self.config.eos_token_id  # 假设config中有eos_token_id
            for b in range(batch_size):
                eos_positions = eos_mask[b].nonzero(as_tuple=True)[0]
                if len(eos_positions) == 0:
                    continue
                positions = torch.arange(seq_len, device=labels.device)
                # 使用searchsorted找到每个位置之后第一个eos的位置
                idx = torch.searchsorted(eos_positions, positions, side='left')
                valid = idx < len(eos_positions)
                next_eos[b, valid] = eos_positions[idx[valid]]

            # 生成三维mask矩阵
            i_indices = torch.arange(seq_len, device=labels.device).view(1, -1, 1)
            j_indices = torch.arange(seq_len, device=labels.device).view(1, 1, -1)
            i_expanded = i_indices.expand(batch_size, -1, -1)
            j_expanded = j_indices.expand(batch_size, -1, -1)

            base_mask = j_expanded > i_expanded  # j必须大于i
            next_eos_expanded = next_eos.unsqueeze(-1).expand(-1, -1, seq_len)
            is_eos_expanded = eos_mask.unsqueeze(-1).expand(-1, -1, seq_len)

            # 处理非eos位置的mask
            mask_non_eos = base_mask & (j_expanded <= next_eos_expanded) & ~is_eos_expanded
            # 处理eos位置的mask（只预测下一个token）
            mask_eos = (j_expanded == i_expanded + 1) & is_eos_expanded & (j_expanded < seq_len)
            mask_eos &= labels.unsqueeze(1) != self.config.eos_token_id  # 排除连续的eos

            final_mask = mask_non_eos | mask_eos

            self._last_mask = final_mask
            # 计算有效位置的损失
            batch_idx, i_idx, j_idx = torch.where(final_mask)
            # print(batch_idx.numel()) # 1569792 , equals to 512*511/2*12 20230123 04:34 found the reason, I was using gpt2 tokenizer, whose eos_token_id is 50256, not equal to geoneox's 0. Now I have changed it. After changing it, the print out value is around 225000
            # print(batch_idx.size(0))
            # 在原有代码基础上修改损失计算部分
            if batch_idx.numel() > 0:
                # 随机采样最多2^15个位置
                num_samples = min(32768, batch_idx.size(0)) #2^15 58GB  (2^14 48GB)
                indices = torch.randperm(batch_idx.size(0), device=lm_logits.device)[:num_samples]
                
                # 获取采样后的数据
                sampled_batch = batch_idx[indices]
                sampled_i = i_idx[indices]
                sampled_j = j_idx[indices]
                
                # 计算采样损失（需调整损失权重）
                selected_logits = lm_logits[sampled_batch, sampled_i, :]
                selected_labels = labels[sampled_batch, sampled_j]
                
                loss_fct = CrossEntropyLoss()
                mtp_loss = loss_fct(selected_logits, selected_labels)
                
                # # 调整损失权重以保持梯度平衡
                # mtp_loss = mtp_loss * (batch_idx.size(0) / num_samples)  # 关键补偿因子
            else:
                mtp_loss = torch.tensor(0.0, device=lm_logits.device)


            lm_loss = ntp_loss + mtp_loss
        # lm_loss = ntp_loss

        if not return_dict:
            output = (ntp_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=ntp_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

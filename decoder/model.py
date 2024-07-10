import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import torch.distributions as t_dist
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.activations import ACT2FN
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.utils import ExplicitEnum


from transformers import GPT2PreTrainedModel, GPT2Config, GenerationConfig, GPT2LMHeadModel
from transformers.generation.utils import GenerationMode
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


class AttnActGPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings+1    # for activation
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        assert self.head_dim * self.num_heads == self.embed_dim

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _upcast_and_reordered_attn(self, query, key, value):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)


        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        activation: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        if activation is not None:
            hidden_states = torch.cat([activation.unsqueeze(1), hidden_states], dim=1)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        if activation is not None:
            query = query[:, 1:]
 
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            present = (key.detach(), value.detach())
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output = self._upcast_and_reordered_attn(query, key, value)
        else:
            attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present



class AttnActGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = AttnActGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

        self.act_proc_layer_block = nn.Sequential(
            Conv1D(hidden_size, config.act_proc_resid_dim),
            nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        activation: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        if activation is not None:
            activation = self.act_proc_layer_block(activation)

        attn_outputs = self.attn(
            hidden_states,
            activation,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,)

        return outputs  # hidden_states, present


class CrossAttnActGPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.act_dim = config.act_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        assert self.head_dim * self.num_heads == self.embed_dim

        self.c_attn_q = Conv1D(self.embed_dim, self.embed_dim)
        self.c_attn_kv = Conv1D(self.embed_dim*2, config.act_proc_resid_dim)
        self.no_op_k = nn.Parameter(
            torch.zeros(1, self.num_heads, 1, self.head_dim)
        )
        self.no_op_v = nn.Parameter(
            torch.zeros(1, self.num_heads, 1, self.head_dim)
        )
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        # attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        activation: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        query = self.c_attn_q(hidden_states)
        query = self._split_heads(query, self.num_heads, self.head_dim)

        if layer_past is None:
            key, value = self.c_attn_kv(activation.unsqueeze(1)).split(self.split_size, dim=-1)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)
            batch_size = key.size(0)
            key = torch.cat([self.no_op_k.expand(batch_size, -1, -1, -1), key], dim=-2)
            value = torch.cat([self.no_op_v.expand(batch_size, -1, -1, -1), value], dim=-2)
        else:
            key, value = layer_past

        if use_cache:
            present = (key.detach(), value.detach())
        else:
            present = None

        attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present


class CrossAttnActGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = AttnActGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

        self.act_proc_layer_ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.act_proc_layer_attn = CrossAttnActGPT2Attention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        activation: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        
        if layer_past is None:
            layer_past1, layer_past2 = None, None
        else:
            layer_past1, layer_past2 = layer_past

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            None,
            layer_past=layer_past1,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present
        outputs = attn_outputs[1]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.act_proc_layer_ln(hidden_states)
        attn_outputs = self.act_proc_layer_attn(
            hidden_states,
            activation,
            layer_past=layer_past2,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present
        outputs = (outputs, attn_outputs[1])
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states, outputs)
        else:
            outputs = (hidden_states,)

        return outputs  # hidden_states, (present1, present2)



class AttnActGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        if config.cross_attn:
            self.h = nn.ModuleList([CrossAttnActGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        else:
            self.h = nn.ModuleList([AttnActGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        activation: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple:
        assert position_ids.size() == input_ids.size()

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        presents = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            outputs = block(
                hidden_states,
                activation=activation,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, presents,
    

class CustomGPT2LMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"act_proc_layer"]

    def __init__(self, config, act_proc_cfg: dict =None):
        super().__init__(config)
        if act_proc_cfg is not None:
            for k, v in act_proc_cfg.items():
                assert not hasattr(config, k)
                setattr(config, k, v)

        self.probed_acts = config.probed_acts
        self.cross_attn = config.cross_attn
        self.act_proc_num_mlp = getattr(config, "act_proc_num_mlp", 6)
        self.prob_weight = OrderedDict({a: 0.0 for a in self.probed_acts})

        self.transformer = AttnActGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.act_proc_layer_emb = nn.Embedding(len(self.probed_acts), config.act_proc_resid_dim-config.act_dim)
        
        self.act_proc_layer_MLPs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.act_proc_resid_dim, eps=config.layer_norm_epsilon),
                Conv1D(config.act_proc_mid_dim, config.act_proc_resid_dim),
                nn.ReLU(), 
                Conv1D(config.act_proc_resid_dim, config.act_proc_mid_dim),
                )
                for i in range(self.act_proc_num_mlp)
        ])
        self.act_proc_layer_final_ln = nn.LayerNorm(config.act_proc_resid_dim, eps=config.layer_norm_epsilon)
            

        # Initialize weights and apply final processing
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        activation: Optional[torch.FloatTensor] = None,
        act_site_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple:
        
        batch_size, seq_len = input_ids.size()
        if past_key_values is None:
            past_length = 0
        elif not self.cross_attn:
            past_length = past_key_values[0][0].size(-2) - 1 # because it contains k/v for activation
        else:
            past_length = past_key_values[0][0][0].size(-2)
        position_ids = torch.arange(past_length, past_length+seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        if activation is not None:
            activation = torch.cat([activation, self.act_proc_layer_emb(act_site_ids)], dim=-1)

            for act_proc_mlp in self.act_proc_layer_MLPs:
                activation = act_proc_mlp(activation) + activation    
            activation = self.act_proc_layer_final_ln(activation)   

        transformer_outputs = self.transformer(
            input_ids,
            position_ids,
            activation=activation,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    @torch.no_grad()
    def generate(
        self,
        activation: Optional[torch.FloatTensor] = None,
        act_site_ids: Optional[torch.LongTensor] = None, 
        generation_mode: Optional[str] = None,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        prev_freq: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.LongTensor]:
        bos_token_id = self.config.bos_token_id
        pad_token_id = self.config.pad_token_id
        eos_token_id = self.config.eos_token_id

        assert activation is not None

        batch_size = activation.size(0)
        assert activation.dim() == 2
        assert activation.size(0) == act_site_ids.size(0)

        inputs = {
            "input_ids": torch.full((batch_size, 1), bos_token_id, device=self.device),
            "activation": activation,
            "act_site_ids": act_site_ids,
            "use_cache": True,
        }

        generated_tokens = inputs["input_ids"]
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if max_len is None:
            max_len = self.config.max_position_embeddings-1
        for i in range(max_len):
            tkn_logits, presents = self(**inputs)
            if generation_mode == "greedy":
                next_tkn_id = tkn_logits.argmax(dim=-1)
            elif generation_mode == "sample":
                if prev_freq is None or (i+1 >= prev_freq.size(0)):
                    next_tkn_id = t_dist.Categorical(logits=tkn_logits / temperature).sample()
                else:
                    probs = F.softmax(tkn_logits / temperature, dim=-1) * (1 - prev_freq[i+1]).view(1, 1, -1)
                    next_tkn_id = t_dist.Categorical(probs=probs).sample()

            next_tkn_id.masked_fill_(terminated.unsqueeze(1), pad_token_id)

            inputs = {
                "input_ids": next_tkn_id,
                "past_key_values": presents,
                "use_cache": True
            }

            generated_tokens = torch.hstack([generated_tokens, next_tkn_id])

            terminated = terminated | (next_tkn_id.squeeze(1) == eos_token_id)
            if terminated.all():
                break

        return generated_tokens


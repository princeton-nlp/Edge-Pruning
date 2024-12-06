# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    ModelOutput
)
from transformers import LlamaConfig, AutoTokenizer
from dataclasses import dataclass

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from l0_fllama import deterministic_z_from_log_alpha, sample_z_from_log_alpha

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class FLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(FLlamaRMSNorm)


class FLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FLlamaLinearScalingRotaryEmbedding(FLlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class FLlamaDynamicNTKScalingRotaryEmbedding(FLlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def num_writers(config, with_embedding_nodes=False):
    # The token and position embeddings are writers, if they exist
    n_writers = 1 if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        # Each attention head is a writer, as is the MLP
        n_writers += config.num_attention_heads + 1
    
    return n_writers

def num_readers(config):
    # The number of readers does not depend on whether the model has embedding nodes
    n_readers = 0
    for l in range(config.num_hidden_layers):
        # Each attention head Q/K/V is a reader, as is the MLP
        n_readers += (config.num_attention_heads + 2 * config.num_key_value_heads) + 1
    # There is a final read
    n_readers += 1
    return n_readers

def num_edges(config, with_embedding_nodes=False):
    # If there are embedding nodes, they write to all readers
    n_edges = num_readers(config) if with_embedding_nodes else 0
    for l in range(config.num_hidden_layers):
        # Each attention head writes to this layer's MLP, (MLP + head Q/K/Vs) of future layers and the final read
        n_edges += config.num_attention_heads * (
            1 + 
            (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) + 
            1
        )
        
        # The MLP writes to (MLP + head Q/K/Vs) of future layers and the final read
        n_edges += (config.num_hidden_layers - l - 1) * (config.num_attention_heads + 2 * config.num_key_value_heads + 1) + 1
    
    return n_edges

def num_nodes(config, with_embedding_nodes=False):
    return num_writers(config, with_embedding_nodes)

def writer_idx_to_name(writer_idx, num_layers, num_heads, with_embedding_nodes=False):
    if with_embedding_nodes:
        if writer_idx == 0:
            return "embeds"
        else:
            writer_idx -= 1
    
    layer_idx = writer_idx // (num_heads + 1)
    head_idx = writer_idx % (num_heads + 1)
    if head_idx == num_heads:
        return f"m{layer_idx}"
    else:
        return f"a{layer_idx}.h{head_idx}"

def writer_name_to_idx(name, num_layers, num_heads, with_embedding_nodes=False):
    idx = 0
    if with_embedding_nodes:
        if name == "embeds":
            return 0
        else:
            idx += 1
    if name.startswith("m"):
        layer_idx = int(name[1:])
        idx += layer_idx * (num_heads + 1) + num_heads
    elif name.startswith("a"):
        parts = name.split(".")
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        idx += layer_idx * (num_heads + 1) + head_idx
    else:
        raise ValueError(f"Unrecognized writer name {name}")
    return idx
    
def reader_idx_to_name(reader_idx, num_layers, num_heads, num_key_value_heads):
    layer_idx = reader_idx // (num_heads + 2 * num_key_value_heads + 1)
    head_idx = reader_idx % (num_heads + 2 * num_key_value_heads + 1)
    if layer_idx == num_layers:
        return "resid_post"
    
    if head_idx < num_heads:
        return f"a{layer_idx}.h{head_idx}.q"
    elif head_idx < num_heads + num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads}.k"
    elif head_idx < num_heads + 2 * num_key_value_heads:
        return f"a{layer_idx}.h{head_idx - num_heads - num_key_value_heads}.v"
    else:
        return f"m{layer_idx}"

def get_mask(log_alpha, training=False, threshold_for_deterministic=None, apply_one=False):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask

class FLlamaMLP(nn.Module):
    def __init__(
        self, 
        config: LlamaConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = FLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FLlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FLlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _apply_headwise_linear(self, x, weight, num_heads):
        # x is (num_q_or_kv_heads, batch_size, seq_len, head_dim)
        # weight is (num_q_or_kv_heads * head_dim, hidden_size)
        # num_heads is num_q_or_kv_heads
        _, bsz, seq_len, _ = x.shape
        weight_ = weight.view(num_heads, self.head_dim, self.hidden_size)
        projected = torch.einsum(
            'nbld,nhd->nblh',
            x,
            weight_
        )
        projected = projected.permute(1, 0, 2, 3)     # (batch_size, n_heads, seq_len, head_dim)
        return projected

    def _apply_output_linear(self, x, weight, num_heads):
        # x is (batch_size, num_heads // tp_factor, seq_len, head_dim)
        # weight is (hidden_size, num_heads * head_dim // tp_factor)
        # num_heads is num_heads // tp_factor
        bsz, _, seq_len, _ = x.shape
        weight_ = weight.view(self.hidden_size, num_heads, self.head_dim)
        projected = torch.einsum(
            'bnlh,dnh->nbld',
            x,
            weight_
        )
        return projected


    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        _, bsz, q_len, _ = q_hidden_states.size()

        if self.config.pretraining_tp > 1:
            assert self.num_heads % self.config.pretraining_tp == 0, "Number of heads must be divisible by pretraining_tp"
            assert self.num_key_value_heads % self.config.pretraining_tp == 0, "Number of key value heads must be divisible by pretraining_tp"
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            # Each q_hidden_state will be (num_heads // pretraining_tp, batch_size, seq_len, head_dim)
            # Each query_slice will be (num_heads * head_dim // pretraining_tp, hidden_size)
            q_hidden_states = q_hidden_states.split(self.num_heads // self.config.pretraining_tp, dim=-1)
            query_states = [
                self._apply_headwise_linear(
                    q_hidden_states[i],
                    query_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ] 
            # Each query_state is (batch_size, num_heads // pretraining_tp, seq_len, head_dim)
            query_states = torch.cat(query_states, dim=1)        
            
            # Each k_hidden_state will be (num_key_value_heads // pretraining_tp, batch_size, seq_len, head_dim)
            # Each key_slice will be (num_key_value_heads * head_dim // pretraining_tp, hidden_size)
            k_hidden_states = k_hidden_states.split(self.num_key_value_heads // self.config.pretraining_tp, dim=-1)
            key_states = [
                self._apply_headwise_linear(
                    k_hidden_states[i],
                    key_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ]
            # Each key_state is (batch_size, num_key_value_heads // pretraining_tp, seq_len, head_dim)
            key_states = torch.cat(key_states, dim=1)
            
            # Each v_hidden_state will be (num_key_value_heads // pretraining_tp, batch_size, seq_len, head_dim) 
            # Each value_slice will be (num_key_value_heads * head_dim // pretraining_tp, hidden_size)
            v_hidden_states = v_hidden_states.split(self.num_key_value_heads // self.config.pretraining_tp, dim=-1)
            value_states = [
                self._apply_headwise_linear(
                    v_hidden_states[i],
                    value_slices[i],
                    self.num_heads // self.config.pretraining_tp
                )
                for i in range(self.config.pretraining_tp)
            ]
            # Each value_state is (batch_size, num_key_value_heads // pretraining_tp, seq_len, head_dim)
            value_states = torch.cat(value_states, dim=1)
        else:
            query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
            key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
            value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.num_heads // self.config.pretraining_tp, dim=1)
            o_proj_slices = self.o_proj.weight.split(self.num_heads * self.head_dim // self.config.pretraining_tp, dim=1)
            attn_output = sum([
                self._apply_output_linear(
                    attn_output[i],
                    o_proj_slices[i],
                    self.num_heads // self.config.pretraining_tp
                ) for i in range(self.config.pretraining_tp)
            ])
        else:
            attn_output = self._apply_output_linear(attn_output, self.o_proj.weight, self.num_heads)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FLlamaFlashAttention2(FLlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _apply_output_linear_flash(self, x, weight, num_heads):
        # x is (batch_size, seq_len, num_heads // tp_factor, head_dim)
        # weight is (hidden_size, num_heads * head_dim // tp_factor)
        # num_heads is num_heads // tp_factor
        bsz, seq_len, _, _ = x.shape
        weight_ = weight.view(self.hidden_size, num_heads, self.head_dim)
        projected = torch.einsum(
            'blnh,dnh->nbld',
            x,
            weight_
        )
        return projected

    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        _, bsz, q_len, _ = q_hidden_states.size()

        query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
        key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
        value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        
        # I think attn output is (batch_size, seq_len, num_heads, head_dim) since it got reshaped to (batch_size, seq_len, hidden_size) originally
        attn_output = self._apply_output_linear_flash(attn_output, self.o_proj.weight, self.num_heads)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class FLlamaSdpaAttention(FLlamaAttention):
    # Adapted from FLlamaAttention.forward
    def forward(
        self,
        q_hidden_states: torch.Tensor,
        k_hidden_states: torch.Tensor,
        v_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                q_hidden_states=hidden_states,
                k_hidden_states=hidden_states,
                v_hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        _, bsz, q_len, _ = hidden_states.size()

        query_states = self._apply_headwise_linear(q_hidden_states, self.q_proj.weight, self.num_heads)
        key_states = self._apply_headwise_linear(k_hidden_states, self.k_proj.weight, self.num_key_value_heads)
        value_states = self._apply_headwise_linear(v_hidden_states, self.v_proj.weight, self.num_key_value_heads)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = self._apply_output_linear(attn_output, self.o_proj.weight, self.num_heads)

        return attn_output, None, past_key_value


FLLAMA_ATTENTION_CLASSES = {
    "eager": FLlamaAttention,
    "flash_attention_2": FLlamaFlashAttention2,
    "sdpa": FLlamaSdpaAttention,
}


class FLlamaDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: LlamaConfig, 
        layer_idx: int,
        with_embedding_nodes: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = FLLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = FLlamaMLP(config)
        self.input_layernorm = FLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads        
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.mlp.gate_proj.weight.dtype
        
        writer_offset = 1 if with_embedding_nodes else 0
        self.attn_writer_idx = writer_offset + layer_idx * (self.num_heads + 1)
        self.attn_reader_idx = layer_idx * (self.num_heads + 2 * self.num_kv_heads + 1)
        self.mlp_writer_idx = writer_offset + (layer_idx + 1) * (self.num_heads + 1) - 1
        self.mlp_reader_idx = (layer_idx + 1) * (self.num_heads + 2 * self.num_kv_heads + 1) - 1
        
        self.q_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.k_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.v_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_kv_heads, dtype=self._dtype))
        self.attn_write_log_alphas = nn.Parameter(torch.empty(self.num_heads, dtype=self._dtype))
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        
        attn_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        attn_read_common_mask[:self.attn_writer_idx] = 1
        attn_read_common_mask = attn_read_common_mask.unsqueeze(1)
        self.register_buffer("attn_read_common_mask", attn_read_common_mask)
        
        attn_write_common_mask = F.pad(
            torch.eye(self.num_heads, dtype=torch.float32).to(self._dtype), # eye does not support bfloat16
            (self.attn_writer_idx, self.num_writers - self.attn_writer_idx - self.num_heads, 0, 0)
        )
        self.register_buffer("attn_write_common_mask", attn_write_common_mask)
        
        self.mlp_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.mlp_write_log_alphas = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        
        mlp_read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        mlp_read_common_mask[:self.mlp_writer_idx] = 1
        self.register_buffer("mlp_read_common_mask", mlp_read_common_mask)
        
        mlp_write_common_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        mlp_write_common_mask[self.mlp_writer_idx, 0] = 1
        self.register_buffer("mlp_write_common_mask", mlp_write_common_mask)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self):
        z_q = get_mask(
            self.q_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_q = z_q[:self.attn_writer_idx, :]
        z_k = get_mask(
            self.k_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_k = z_k[:self.attn_writer_idx, :]
        z_v = get_mask(
            self.v_read_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_v = z_v[:self.attn_writer_idx, :]
        
        z_mlp = get_mask(
            self.mlp_read_log_alphas, 
            training=self.training, 
            threshold_for_deterministic=self.edge_threshold_for_deterministic
        )
        z_mlp = z_mlp[:self.mlp_writer_idx]
        
        return (z_q, z_k, z_v, z_mlp)
    
    @torch.no_grad()
    def get_node_masks(self):
        z_attn = get_mask(
            self.attn_write_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.node_threshold_for_deterministic
        )
                
        z_mlp = get_mask(
            self.mlp_write_log_alphas, 
            training=self.training, 
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape([])
        
        return (z_attn, z_mlp)

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.attn_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)

    @torch.no_grad()
    def load_attn_log_alphas(self, attn_in_edges):
        # Fill with -10 by default
        self.q_read_log_alphas.data.fill_(-10)
        self.k_read_log_alphas.data.fill_(-10)
        self.v_read_log_alphas.data.fill_(-10)
        
        for writer_idx, reader in attn_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 3, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            head = int(reader_portions[1][1:])
            qkv = reader_portions[2]
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            if qkv == "q":
                self.q_read_log_alphas[writer_idx, head] = 10
            elif qkv == "k":
                self.k_read_log_alphas[writer_idx, head] = 10
            elif qkv == "v":
                self.v_read_log_alphas[writer_idx, head] = 10
        
        # Fill with 10 for node masks, since we don't want any further restraint on edges
        self.attn_write_log_alphas.data.fill_(10)
    
    @torch.no_grad()
    def load_mlp_log_alphas(self, mlp_in_edges):
        # Fill with -10 by default
        self.mlp_read_log_alphas.data.fill_(-10)
        
        for writer_idx, reader in mlp_in_edges:
            reader_portions = reader.split(".")
            assert len(reader_portions) == 1, f"Invalid reader format: {reader}"
            layer_idx = int(reader_portions[0][1:])
            assert layer_idx == self.layer_idx, f"Invalid layer index: {layer_idx}"
            self.mlp_read_log_alphas[writer_idx] = 10
        
        # Fill with 10 for node masks, since we don't want any further restraint on edges
        self.mlp_write_log_alphas.data.fill_(10)

    def attn_read(self, x, corr_x=None, embeds=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)
        
        q_m = get_mask(self.q_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        k_m = get_mask(self.k_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        v_m = get_mask(self.v_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        
        q_z = q_m * self.attn_read_common_mask
        k_z = k_m * self.attn_read_common_mask
        v_z = v_m * self.attn_read_common_mask
        
        x_q = torch.einsum("wbsd,wh->hbsd", x, q_z)
        x_k = torch.einsum("wbsd,wh->hbsd", x, k_z)
        x_v = torch.einsum("wbsd,wh->hbsd", x, v_z)
        
        if embeds is not None:
            x_q = x_q + embeds.unsqueeze(0)
            x_k = x_k + embeds.unsqueeze(0)
            x_v = x_v + embeds.unsqueeze(0)
        
        if corr_x is not None:
            x_q = x_q + torch.einsum("wbsd,wh->hbsd", corr_x, (1-q_m) * self.attn_read_common_mask)
            x_k = x_k + torch.einsum("wbsd,wh->hbsd", corr_x, (1-k_m) * self.attn_read_common_mask)
            x_v = x_v + torch.einsum("wbsd,wh->hbsd", corr_x, (1-v_m) * self.attn_read_common_mask)
            
        z_edges_sum = torch.sum(q_z) + torch.sum(k_z) + torch.sum(v_z)
        
        return x_q, x_k, x_v, z_edges_sum
    
    def attn_write(self, residual, x, corr_x=None):
        # residual is (writers, batch_size, sequence_length, hidden_size)
        # x is (num_heads, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        z = get_mask(
            self.attn_write_log_alphas, 
            training=self.training, 
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape(-1, 1, 1, 1)
        x = x * z
        
        if corr_x is not None:
            x = x + corr_x[self.attn_writer_idx : self.attn_writer_idx + self.num_heads] * (1-z)
            
        x = torch.einsum("nbsd,nw->wbsd", x, self.attn_write_common_mask)
        
        residual = residual + x
        z_nodes_sum = torch.sum(z)
        
        return residual, z_nodes_sum

    def mlp_read(self, x, corr_x=None, embeds=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)
        m = get_mask(self.mlp_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)

        z = m * self.mlp_read_common_mask
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        
        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-m) * self.mlp_read_common_mask)

        z_edges_sum = torch.sum(z)
        
        return x_z, z_edges_sum

    def mlp_write(self, residual, x, corr_x=None):
        # residual is (writers, batch_size, sequence_length, hidden_size)
        # x is (batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        z = get_mask(
            self.mlp_write_log_alphas, 
            training=self.training, 
            threshold_for_deterministic=self.node_threshold_for_deterministic
        ).reshape(1, 1, 1)
        x = x * z
        
        if corr_x is not None:
            x = x + corr_x[self.mlp_writer_idx] * (1-z)
            
        x = torch.einsum("ibsd,wi->wbsd", x.unsqueeze(0), self.mlp_write_common_mask)
        residual = residual + x
        
        return residual, torch.sum(z)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        corr_x: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # hidden_states is now (writers, batch_size, sequence_length, hidden_size)
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        
        q_hidden_states, k_hidden_states, v_hidden_states, z_attn_edges_sum = self.attn_read(hidden_states, corr_x=corr_x, embeds=embeds)
        q_hidden_states = self.input_layernorm(q_hidden_states)
        k_hidden_states = self.input_layernorm(k_hidden_states)
        v_hidden_states = self.input_layernorm(v_hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            q_hidden_states=q_hidden_states,
            k_hidden_states=k_hidden_states,
            v_hidden_states=v_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        residual, z_attn_nodes_sum = self.attn_write(residual, hidden_states, corr_x=corr_x)
        
        # Fully Connected
        hidden_states, z_mlp_edges_sum = self.mlp_read(residual, corr_x=corr_x, embeds=embeds)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states, z_mlp_nodes_sum = self.mlp_write(residual, hidden_states, corr_x=corr_x)
        
        z_edges_sum = z_attn_edges_sum + z_mlp_edges_sum
        z_nodes_sum = z_attn_nodes_sum + z_mlp_nodes_sum

        # print_str = f"Layer {self.layer_idx}: {str(hidden_states.device)} {str(torch.cuda.max_memory_allocated() / 1024**3)} GB"
        # print(print_str)

        outputs = (hidden_states, z_edges_sum, z_nodes_sum)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class FLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None

@dataclass 
class FLlamaModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None

class FLlamaModel(FLlamaPreTrainedModel):
    def __init__(
        self, 
        config: LlamaConfig,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                FLlamaDecoderLayer(config, layer_idx, with_embedding_nodes=with_embedding_nodes) 
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads        
        self.num_writers = num_writers(config, with_embedding_nodes=with_embedding_nodes)
        self.num_readers = num_readers(config)
        self.num_layers = config.num_hidden_layers
        self.num_edges = num_edges(config, with_embedding_nodes=with_embedding_nodes)
        self.num_nodes = num_nodes(config, with_embedding_nodes=with_embedding_nodes)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = self.norm.weight.dtype
        self.with_embedding_nodes = with_embedding_nodes
        
        if self.with_embedding_nodes:
            self.token_write_log_alpha = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
            
            token_write_mask = torch.zeros(self.num_writers, dtype=self._dtype)
            token_write_mask[0] = 1
            self.register_buffer("token_write_mask", token_write_mask)
        
        self.final_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        
        if disable_linear_regularization_term:
            self.sparsity_lambda_edges_1 = torch.tensor([0.0], dtype=self._dtype)
            self.sparsity_lambda_nodes_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_edges_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
            self.sparsity_lambda_nodes_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_edges_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_nodes_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        for layer in self.layers:
            layer.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)
    
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic
        for layer in self.layers:
            layer.set_node_threshold_for_deterministic(node_threshold_for_deterministic)
            
    @torch.no_grad()
    def get_edge_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_edge_masks())
        z_final = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        masks.append((z_final,))
        return masks
    
    @torch.no_grad()
    def get_node_masks(self):
        masks = []
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha, 
                training=self.training, 
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape([])
            masks.append((z_tokens,))
        for layer in self.layers:
            masks.append(layer.get_node_masks())
        return masks
    
    @torch.no_grad()
    def get_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_
        
        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_
        
        s /= (1 if n == 0 else n)
        return 1 - s
    
    @torch.no_grad()
    def get_node_sparsity(self):
        node_masks = self.get_node_masks()
        def process(mask):
            return torch.sum(mask), torch.numel(mask)
        s, n = 0, 0
        if self.with_embedding_nodes:
            s_, n_ = process(node_masks[0][0])
            s += s_
            n += n_
            offset = 1
        else:
            offset = 0
        for l in range(len(self.layers)):
            for i in range(2):
                s_, n_ = process(node_masks[l+offset][i])
                s += s_
                n += n_
        
        s /= (1 if n == 0 else n)
        return 1 - s
    
    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()
        
        full_node_mask = torch.cat([mask.reshape(-1) for group in node_masks for mask in group], dim=0)
        
        def process(mask):
            mask = mask * full_node_mask[:mask.shape[0]].reshape(-1, *([1] * (mask.ndim - 1)))
            return torch.sum(mask), torch.numel(mask)
        
        s, n = 0, 0
        for l in range(self.num_layers):
            for i in range(4):
                s_, n_ = process(edge_masks[l][i])
                s += s_
                n += n_
        
        s_, n_ = process(edge_masks[-1][0])
        s += s_
        n += n_
        
        s /= (1 if n == 0 else n)
        return 1 - s        
    
    @torch.no_grad()
    def get_edges(self):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()
        
        allowed_writers = []
        edges = []
        
        if self.with_embedding_nodes:
            if node_masks[0][0] == 1:
                allowed_writers.append(0)
            offset = 1
            layer_offset = 1
        else:
            offset = 0
            layer_offset = 0
        
        for l in range(self.num_layers):
            attn_writers = node_masks[l+layer_offset][0]
            for i in range(self.num_heads):
                if attn_writers[i] == 1:
                    allowed_writers.append(offset + l * (1 + self.num_heads) + i)
            mlp_writers = node_masks[l+layer_offset][1]
            if mlp_writers == 1:
                allowed_writers.append(offset + (l+1) * (1 + self.num_heads) - 1)
        
            attn_q_edges, attn_k_edges, attn_v_edges, mlp_edges = edge_masks[l]
            for from_idx in range(attn_q_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                for head_no in range(attn_q_edges.shape[1]):
                    if attn_q_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes), 
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_k_edges.shape[1]):
                    if attn_k_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes), 
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
                for head_no in range(attn_v_edges.shape[1]):
                    if attn_v_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + self.num_heads + 2 * self.num_kv_heads) + self.num_heads + self.num_kv_heads + head_no
                        edges.append((
                            writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes), 
                            reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                        ))
            for from_idx in range(mlp_edges.shape[0]):
                if from_idx not in allowed_writers:
                    continue
                if mlp_edges[from_idx] == 1:
                    to_idx = (l+1) * (1 + self.num_heads + 2 * self.num_kv_heads) - 1
                    edges.append((
                        writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes), 
                        reader_idx_to_name(to_idx, num_layers=self.num_layers, num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads)
                    ))
        final_read_mask = edge_masks[self.num_layers][0]
        for from_idx in range(self.num_writers):
            if (from_idx in allowed_writers) and (final_read_mask[from_idx] == 1):
                edges.append((
                    writer_idx_to_name(from_idx, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes), 
                    f"resid_post"
                ))
        return edges
    
    @torch.no_grad()
    def reset_all_log_alphas(self):
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        for layer in self.layers:
            layer.reset_all_log_alphas()
        self.final_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.sparsity_lambda_edges_1.data.zero_()
        self.sparsity_lambda_nodes_1.data.zero_()
    
    @torch.no_grad()
    def load_resid_post_log_alphas(self, edges):
        # Fill with -10 by default
        self.final_read_log_alphas.data.fill_(-10)
        
        for writer_idx, reader in edges:
            assert reader == "resid_post", f"Invalid reader format: {reader}"
            self.final_read_log_alphas[writer_idx] = 10
        
        # Fill with 10 for node masks, since we don't want any further restraint on edges
        if self.with_embedding_nodes:
            self.token_write_log_alpha.data.fill_(10)
    
    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        layer_attn_in_edges = [[] for _ in range(self.num_layers)]
        layer_mlp_in_edges = [[] for _ in range(self.num_layers)]
        resid_post_edges = []
        for edge in edges:
            writer, reader = edge
            writer_idx = writer_name_to_idx(writer, num_layers=self.num_layers, num_heads=self.num_heads, with_embedding_nodes=self.with_embedding_nodes)
            if reader == "resid_post":
                resid_post_edges.append((writer_idx, reader))
            elif reader.startswith("m"):
                layer_idx = int(reader[1:])
                layer_mlp_in_edges[layer_idx].append((writer_idx, reader))
            elif reader.startswith("a"):
                layer_idx = int(reader[1:reader.find(".")])
                layer_attn_in_edges[layer_idx].append((writer_idx, reader))
            else:
                raise ValueError(f"Invalid reader format: {reader}")
        for layer_idx, attn_in_edges in enumerate(layer_attn_in_edges):
            self.layers[layer_idx].load_attn_log_alphas(attn_in_edges)
        for layer_idx, mlp_in_edges in enumerate(layer_mlp_in_edges):
            self.layers[layer_idx].load_mlp_log_alphas(mlp_in_edges)
        self.load_resid_post_log_alphas(resid_post_edges)
    
    def read(self, x, corr_x=None, embeds=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        # embeds, if it exists, is (batch_size, sequence_length, hidden_size)
        z = get_mask(self.final_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        
        if embeds is not None:
            x_z = x_z + embeds
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-z))
            
        z_edges_sum = torch.sum(z)
        
        return x_z, z_edges_sum
    
    def write(self, tok_embeds, corr_x=None):
        # tok_embeds is (batch_size, sequence_length, hidden_size)
        # corr_x, if it exists, is (writers, batch_size, sequence_length, hidden_size)
        if self.with_embedding_nodes:
            z_tokens = get_mask(
                self.token_write_log_alpha, 
                training=self.training, 
                threshold_for_deterministic=self.node_threshold_for_deterministic
            ).reshape(1, 1, 1)
            tok_embeds = tok_embeds * z_tokens
            if corr_x is not None:
                tok_embeds = tok_embeds + corr_x[0] * (1 - z_tokens)
            
            # hidden_states = tok_embeds.unsqueeze(0) * self.token_write_mask.reshape(-1, 1, 1, 1)    
            hidden_states = tok_embeds.detach().unsqueeze(0) * self.token_write_mask.reshape(-1, 1, 1, 1)      
            z_nodes_sum = torch.sum(z_tokens)
            
            return hidden_states, None, z_nodes_sum
        else:
            hidden_states = torch.zeros(self.num_writers, *tok_embeds.shape, dtype=tok_embeds.dtype, device=tok_embeds.device)
            z_nodes_sum = 0
            return hidden_states, tok_embeds, z_nodes_sum
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x = None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FLlamaModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens + inputs_embeds.shape[1]
        )

        # embed positions
        hidden_states, embeds, z_nodes_sum = self.write(inputs_embeds, corr_x=corr_x)
        z_edges_sum = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    corr_x,
                    embeds,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    corr_x=corr_x,
                    embeds=embeds,
                )

            hidden_states, z_layer_edges_sum, z_layer_nodes_sum = layer_outputs[0], layer_outputs[1], layer_outputs[2]
            z_edges_sum = z_edges_sum + z_layer_edges_sum
            z_nodes_sum = z_nodes_sum + z_layer_nodes_sum

            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        if output_writer_states:
            writer_states = hidden_states
        else:
            writer_states = None
            
        hidden_states, z_final_edges_sum = self.read(hidden_states, corr_x=corr_x, embeds=embeds)
        z_edges_sum = z_edges_sum + z_final_edges_sum
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,) 

        model_edge_sparsity = 1 - (z_edges_sum / self.num_edges)
        model_node_sparsity = 1 - (z_nodes_sum / self.num_nodes)
        
        if target_edge_sparsity is None:
            edge_loss = None
        else:
            edge_loss = self.sparsity_lambda_edges_1.reshape([]) * (
                model_edge_sparsity - target_edge_sparsity
            ) + self.sparsity_lambda_edges_2.reshape([]) * (
                model_edge_sparsity - target_edge_sparsity
            )**2
            
        if target_node_sparsity is None:
            node_loss = None
        else:
            node_loss = self.sparsity_lambda_nodes_1.reshape([]) * (
                model_node_sparsity - target_node_sparsity
            ) + self.sparsity_lambda_nodes_2.reshape([]) * (
                model_node_sparsity - target_node_sparsity
            )**2

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        
        if target_edge_sparsity is not None:
            target_edge_sparsity = torch.tensor(target_edge_sparsity, device=model_edge_sparsity.device, dtype=model_edge_sparsity.dtype)
        if target_node_sparsity is not None:
            target_node_sparsity = torch.tensor(target_node_sparsity, device=model_node_sparsity.device, dtype=model_node_sparsity.dtype)
        
        if not return_dict:
            return tuple(
                v 
                for v in [
                    hidden_states, 
                    next_cache, 
                    all_hidden_states, 
                    all_self_attns,
                    writer_states,
                    target_edge_sparsity,
                    target_node_sparsity,
                    model_edge_sparsity,
                    model_node_sparsity,
                    edge_loss,
                    node_loss,
                ] if v is not None
            )
        return FLlamaModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            writer_states=writer_states,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            model_edge_sparsity=model_edge_sparsity,
            model_node_sparsity=model_node_sparsity,
            edge_loss=edge_loss,
            node_loss=node_loss,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, current_length):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else current_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

@dataclass 
class FLlamaForCausalLMOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None

class FLlamaForCausalLM(FLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self, 
        config: LlamaConfig,
        with_embedding_nodes: bool = False,
        disable_linear_regularization_term=False,
    ):
        super().__init__(config)
        self.model = FLlamaModel(
            config,
            with_embedding_nodes=with_embedding_nodes,
            disable_linear_regularization_term=disable_linear_regularization_term 
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.model.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)
    
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.model.set_node_threshold_for_deterministic(node_threshold_for_deterministic)
        
    @torch.no_grad()
    def get_edge_masks(self):
        return self.model.get_edge_masks()
    
    @torch.no_grad()
    def get_node_masks(self):
        return self.model.get_node_masks()
    
    @torch.no_grad()
    def get_edge_sparsity(self):
        return self.model.get_edge_sparsity()
    
    @torch.no_grad()
    def get_node_sparsity(self):
        return self.model.get_node_sparsity()
    
    @torch.no_grad()
    def get_effective_edge_sparsity(self):
        return self.model.get_effective_edge_sparsity()
    
    @torch.no_grad()
    def get_edges(self):
        return self.model.get_edges()

    @torch.no_grad()
    def reset_all_log_alphas(self):
        self.model.reset_all_log_alphas()
        
    @torch.no_grad()
    def load_all_log_alphas(self, edges):
        self.model.load_all_log_alphas(edges)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_edge_sparsity: Optional[float] = None,
        target_node_sparsity: Optional[float] = None,
        corr_x = None,
        output_writer_states: Optional[bool] = False,
    ) -> Union[Tuple, FLlamaForCausalLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            corr_x=corr_x,
            output_writer_states=output_writer_states,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FLlamaForCausalLMOutput(
            lm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            writer_states=outputs.writer_states,
            target_edge_sparsity=outputs.target_edge_sparsity,
            target_node_sparsity=outputs.target_node_sparsity,
            model_edge_sparsity=outputs.model_edge_sparsity,
            model_node_sparsity=outputs.model_node_sparsity,
            edge_loss=outputs.edge_loss,
            node_loss=outputs.node_loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

if __name__ == '__main__':
    model = FLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, with_embedding_nodes=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    inputs = tokenizer("Hi, I am John. I", return_tensors="pt").to("cuda")

    model.train()
    logits = model(**inputs).logits

    # model.set_edge_threshold_for_deterministic(0.0)
    # model.set_node_threshold_for_deterministic(0.0)
    
    # outputs = model.generate(**inputs, max_length=30)
    # output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # print(output)
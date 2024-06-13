# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch Erazr model."""

import math
import os
import warnings
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import PretrainedConfig

import numpy as np
from l0 import deterministic_z_from_log_alpha, sample_z_from_log_alpha

logger = logging.get_logger(__name__)

@dataclass
class ErazrConfig(PretrainedConfig):
    model_size: int = 41
    num_layers: int = 4
    num_heads: int = 1
    key_size: int = 12
    value_size: int = 12
    mlp_hidden_size: int = 30
    vocab_size: int = 5         # The second-last is "BOS", and last is "compiler_pad" 
    max_position_embeddings: int = 5
    activation_function: str = "relu"
    layer_norm: bool = True
    causal: bool = True
    dtype: str = "fp32"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

dtype_to_torch = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def format_weight_dict(weight_dict, n_layers):
    tok_embed = weight_dict["token_embed_embeddings"]
    pos_embed = weight_dict["pos_embed_embeddings"]
    blocks_embeds = []
    for l in range(n_layers):
        k_w = weight_dict[f"transformer/layer_{l}/attn/key_w"]
        k_b = weight_dict[f"transformer/layer_{l}/attn/key_b"]
        q_w = weight_dict[f"transformer/layer_{l}/attn/query_w"]
        q_b = weight_dict[f"transformer/layer_{l}/attn/query_b"]
        v_w = weight_dict[f"transformer/layer_{l}/attn/value_w"]
        v_b = weight_dict[f"transformer/layer_{l}/attn/value_b"]
        o_w = weight_dict[f"transformer/layer_{l}/attn/linear_w"]
        o_b = weight_dict[f"transformer/layer_{l}/attn/linear_b"]
        up_w = weight_dict[f"transformer/layer_{l}/mlp/linear_1_w"]
        up_b = weight_dict[f"transformer/layer_{l}/mlp/linear_1_b"]
        down_w = weight_dict[f"transformer/layer_{l}/mlp/linear_2_w"]
        down_b = weight_dict[f"transformer/layer_{l}/mlp/linear_2_b"]
        
        blocks_embeds.append([q_w, k_w, v_w, o_w, up_w, down_w, q_b, k_b, v_b, o_b, up_b, down_b])
    return tok_embed, pos_embed, blocks_embeds

def get_key_value_vocab_and_model_size(tok_embed, blocks_embeds, num_heads):
    key_size = blocks_embeds[0][0].shape[1] // num_heads
    value_size = blocks_embeds[0][2].shape[1] // num_heads
    vocab_size = tok_embed.shape[0]
    model_size = tok_embed.shape[1]
    return key_size, value_size, vocab_size, model_size

def get_config_weights_and_vocab(input_path, pytorch=True, device=None, act="relu"):
    config_and_weights = pickle.load(open(input_path, "rb"))
    config_ = config_and_weights["config"]
    tok_embed, pos_embed, blocks_embeds = format_weight_dict(config_and_weights["model_params"], config_["num_layers"])
    unembedding_mtx = None if "unembedding_mtx" not in config_and_weights else config_and_weights["unembedding_mtx"]
    key_size, value_size, vocab_size, model_size = get_key_value_vocab_and_model_size(tok_embed, blocks_embeds, config_["num_heads"])

    config = ErazrConfig(
        model_size=model_size,
        num_layers=config_["num_layers"],
        num_heads=config_["num_heads"],
        key_size=key_size,
        value_size=value_size,
        mlp_hidden_size=config_["mlp_hidden_size"],
        vocab_size=vocab_size,
        max_position_embeddings=config_["max_seq_len"],
        activation_function=act,
        layer_norm=config_["layer_norm"],
        causal=config_["causal"],
    )

    if pytorch:
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        tok_embed = torch.tensor(tok_embed).to(device)
        pos_embed = torch.tensor(pos_embed).to(device)
        unembedding_mtx = None if unembedding_mtx is None else torch.tensor(unembedding_mtx).to(device=device, dtype=tok_embed.dtype)
        blocks_embeds = [
            [torch.tensor(w).to(device) for w in block]
            for block in blocks_embeds
        ]
    return config, tok_embed, pos_embed, blocks_embeds, config_and_weights["vocab"], config_["bos"], config_["pad"], unembedding_mtx

def n_readers(config):
    # MLP + Q/K/V per head for each block, plus final read
    return config.num_layers * (1 + config.num_heads * 3) + 1

def n_writers(config):
    # the two embeddings, MLP + each head in every block
    return 2 + config.num_layers * (1 + config.num_heads)

def n_total_edges(config):
    n_edges = 2 * n_readers(config) # the token and positional embeddings write to every reader
    for i in range(config.num_layers):
        # Each attention head writes to the MLP in the same layer, as well as every node in future layers (including the final read)
        n_edges += config.num_heads * (1 + (1 + 3 * config.num_heads) * (config.num_layers - i - 1) + 1)
        # Each MLP writes to every node in future layers (including the final read)
        n_edges += (1 + 3 * config.num_heads) * (config.num_layers - i - 1) + 1
    return n_edges

def get_writer_name(idx, num_heads):
    if idx == 0:
        return "token_embeds"
    elif idx == 1:
        return "pos_embeds"
    idx -= 2
    layer = idx // (1 + num_heads)
    rem = idx % (1 + num_heads)
    if rem == num_heads:
        return f"m{layer}"
    else:
        return f"a{layer}.h{rem}"
    
def get_reader_name(idx, num_heads):
    layer = idx // (1 + 3 * num_heads)
    rem = idx % (1 + 3 * num_heads)
    rem_rem = rem % num_heads
    if rem < num_heads:
        return f"a{layer}.h{rem_rem}.q"
    elif rem < 2 * num_heads:
        return f"a{layer}.h{rem_rem}.k"
    elif rem < 3 * num_heads:
        return f"a{layer}.h{rem_rem}.v"
    else:
        return f"m{layer}"

def get_mask(log_alpha, training=False, threshold_for_deterministic=None, apply_one=False):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask

class ErazrTokenizer:
    def __init__(self, vocab, bos, pad):
        self.vocab = vocab
        self.bos = bos
        self.pad = pad
        self.vocab_dict = vocab
        self.rev_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
        self.bos_token = self.vocab_dict[self.bos]
        self.pad_token = self.vocab_dict[self.pad]

    def pad_or_truncate(tokens, max_length, padding=False, truncate=False):
        if truncate and len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif padding and len(tokens) < max_length:
            tokens = tokens + [self.vocab_dict[self.pad]] * (max_length - len(tokens))
        return tokens

    def encode_single(self, text, max_length=None, padding=False, truncate=False, return_tensors=False, add_special_tokens=True):
        if type(text) != list:
            text = text.split()
        if add_special_tokens and (len(text) == 0 or text[0] != self.bos):
            tokens = [self.bos_token]
        else:
            tokens = []
        tokens += [self.vocab_dict.get(t, self.vocab_dict[self.pad]) for t in text]
        if max_length is not None:
            tokens = self.pad_or_truncate(tokens, max_length, padding, truncate)
        if return_tensors == "np":
            return np.array(tokens, dtype=int)
        elif return_tensors == "pt":
            return torch.LongTensor(tokens)
        else:
            return tokens

    def encode(self, texts, max_length=None, padding=False, truncate=False, return_tensors=False, add_special_tokens=True):
        if type(texts) == str:
            texts = [texts]
        encoded = [self.encode_single(t, max_length, padding, truncate, add_special_tokens=add_special_tokens) for t in texts]
        max_l = max([len(e) for e in encoded])
        min_l = min([len(e) for e in encoded])
        if min_l != max_l:
            assert padding, "All sequences must have the same length if padding is not enabled."
            encoded = [self.pad_or_truncate(e, max_l, padding, truncate) for e in encoded]
        if return_tensors == "np":
            return np.array(encoded, dtype=int)
        elif return_tensors == "pt":
            return torch.LongTensor(encoded)
        else:
            return encoded

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode_single(self, tokens, remove_special_tokens=True, starts_with_bos=False):
        if type(tokens) == torch.Tensor or type(tokens) == np.ndarray:
            tokens = tokens.tolist()
        if starts_with_bos:     # The bos token is somehow not in the unembedding matrix
            tokens[0] = self.bos_token
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in [self.bos_token, self.pad_token]]
        return [self.rev_vocab_dict[t] for t in tokens]        
    
    def decode(self, tokens, remove_special_tokens=True, starts_with_bos=False):
        return [self.decode_single(t, remove_special_tokens, starts_with_bos) for t in tokens]

class ErazrAttention(nn.Module):
    def __init__(
        self, 
        config: ErazrConfig,
        reader_idx: int,
        writer_idx: int
    ):
        super().__init__()
        self.num_heads = config.num_heads
        self.key_size = config.key_size
        self.value_size = config.value_size
        self.model_size = config.model_size
        self.reader_idx = reader_idx
        self.writer_idx = writer_idx
        self.num_readers = n_readers(config)
        self.num_writers = n_writers(config)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = dtype_to_torch[config.dtype]
        
        if config.layer_norm:
            self.input_layernorm = nn.LayerNorm(self.model_size)
        else:
            self.input_layernorm = nn.Identity()
        
        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.w_o = None
        self.b_q = None
        self.b_k = None
        self.b_v = None
        self.b_o = None
        
        self.initialized = False
        
        self.q_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.k_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.v_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, self.num_heads, dtype=self._dtype))
        self.write_log_alphas = nn.Parameter(torch.empty(self.num_heads, dtype=self._dtype))
        
        self.q_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.k_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.v_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.write_log_alphas.data.normal_(mean=10.0, std=0.01)
        
        read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        read_common_mask[:self.writer_idx] = 1
        read_common_mask = read_common_mask.reshape(-1, 1)
        self.register_buffer("read_common_mask", read_common_mask)
        
        write_common_mask = F.pad(
            torch.eye(self.num_heads, dtype=self._dtype),
            (self.writer_idx, self.num_writers - (self.writer_idx + self.num_heads), 0, 0),
        )
        self.register_buffer("write_common_mask", write_common_mask)

    def load_w_qkv(self, w_q, w_k, w_v, w_o, b_q=None, b_k=None, b_v=None, b_o=None):
        self.w_q = w_q.reshape(self.model_size, self.num_heads, -1)
        self.w_k = w_k.reshape(self.model_size, self.num_heads, -1)
        self.w_v = w_v.reshape(self.model_size, self.num_heads, -1)
        self.w_o = w_o.reshape(self.num_heads, -1, self.model_size)
        self.b_q = None if b_q is None else b_q.reshape(1, 1, self.num_heads, -1)
        self.b_k = None if b_k is None else b_k.reshape(1, 1, self.num_heads, -1)
        self.b_v = None if b_v is None else b_v.reshape(1, 1, self.num_heads, -1)
        self.b_o = None if b_o is None else b_o.reshape(1, 1, 1, -1).repeat(self.num_heads, 1, 1, 1) / self.num_heads
        self.initialized = True

    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
    
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self):
        z_q = get_mask(self.q_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_q = z_q[:self.writer_idx, :]
        z_k = get_mask(self.k_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_k = z_k[:self.writer_idx, :]
        z_v = get_mask(self.v_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_v = z_v[:self.writer_idx, :]
        return (z_q, z_k, z_v)

    @torch.no_grad()
    def get_node_masks(self):
        z = get_mask(self.write_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        return (z,)

    def read(self, x, corr_x=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # embeds are (batch_size, sequence_length, hidden_size) and always contribute
        # corr_x is (writers, batch_size, sequence_length, hidden_size) and only contributes if not None
        # masks are (writers, num_heads) each
        
        q_m = get_mask(self.q_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        k_m = get_mask(self.k_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        v_m = get_mask(self.v_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        
        q_z = q_m * self.read_common_mask
        k_z = k_m * self.read_common_mask
        v_z = v_m * self.read_common_mask
        
        x_q = torch.einsum("wbsd,wh->hbsd", x, q_z)
        x_k = torch.einsum("wbsd,wh->hbsd", x, k_z)
        x_v = torch.einsum("wbsd,wh->hbsd", x, v_z)
        
        if corr_x is not None:
            x_q = x_q + torch.einsum("wbsd,wh->hbsd", corr_x, (1-q_m) * self.read_common_mask)
            x_k = x_k + torch.einsum("wbsd,wh->hbsd", corr_x, (1-k_m) * self.read_common_mask)
            x_v = x_v + torch.einsum("wbsd,wh->hbsd", corr_x, (1-v_m) * self.read_common_mask)
        
        x_q = self.input_layernorm(x_q)
        x_k = self.input_layernorm(x_k)
        x_v = self.input_layernorm(x_v)
        
        z_edges_sum = torch.sum(q_z + k_z + v_z)
        
        return x_q, x_k, x_v, z_edges_sum

    def write(self, residual, x):
        # x is (num_heads, batch_size, sequence_length, hidden_size)
        # residual is (writers, batch_size, sequence_length, hidden_size)
        x = torch.einsum("nbsd,nw->wbsd", x, self.write_common_mask)
        
        residual = residual + x
        return residual

    def apply_proj(self, x, w, kv_size, b=None):
        # x is (num_heads, batch_size, sequence_length, hidden_size)
        # w is (hidden_size, num_heads, kv_size)
        # b is (num_heads, kv_size)
        proj = torch.einsum("nbsd,dnz->bsnz", x, w)
        if b is not None:
            proj = proj + b
        return proj

    def apply_inv_proj(self, x, w, b=None, corr_x=None):
        # x is (batch_size, sequence_length, num_heads, kv_size)
        # w is (num_heads, kv_size, hidden_size)
        # b is (hidden_size,)
        proj = torch.einsum("bnsz,nzd->nbsd", x, w)
        if b is not None:
            proj = proj + b
        
        out_z = get_mask(self.write_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic).reshape(-1, 1, 1, 1)
        if corr_x is not None:
            proj = proj * out_z + corr_x[self.writer_idx : self.writer_idx + self.num_heads] * (1 - out_z)
        else:
            proj = proj * out_z
            
        z_nodes_sum = torch.sum(out_z) 
        
        return proj, z_nodes_sum

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        corr_x=None,
    ):
        residual = hidden_states        # (writers, batch_size, sequence_length, hidden_size)
        
        hidden_q, hidden_k, hidden_v, z_edges_sum = self.read(hidden_states, corr_x=corr_x)
        
        query = self.apply_proj(hidden_q, self.w_q, self.key_size, self.b_q)
        key = self.apply_proj(hidden_k, self.w_k, self.key_size, self.b_k)
        value = self.apply_proj(hidden_v, self.w_v, self.value_size, self.b_v)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_weights = torch.einsum("bqnh,bknh->bnqk", query, key) / math.sqrt(self.key_size)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1) == 0, -1e30)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bnqk,bknh->bnqh", attn_weights, value)
        # attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads * self.value_size)
        attn_output, z_nodes_sum = self.apply_inv_proj(attn_output, self.w_o, self.b_o, corr_x=corr_x)  
        
        residual = self.write(residual, attn_output)

        outputs = (residual, z_edges_sum, z_nodes_sum, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class ErazrMLP(nn.Module):
    def __init__(
        self, 
        config: ErazrConfig,
        reader_idx: int,
        writer_idx: int,
    ):
        super().__init__()
        self.model_size = config.model_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.reader_idx = reader_idx
        self.writer_idx = writer_idx
        self.num_readers = n_readers(config)
        self.num_writers = n_writers(config)
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        self._dtype = dtype_to_torch[config.dtype]
        
        if config.layer_norm:
            self.input_layernorm = nn.LayerNorm(self.model_size)
        else:
            self.input_layernorm = nn.Identity()
        
        self.act = ACT2FN[config.activation_function]
        
        self.up_proj = None
        self.down_proj = None
        self.up_bias = None
        self.down_bias = None
        self.initialized = False
        
        self.mlp_read_log_alphas = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.mlp_write_log_alphas = nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.mlp_read_log_alphas.data.normal_(mean=10.0, std=0.01)
        self.mlp_write_log_alphas.data.normal_(mean=10.0, std=0.01)
        
        read_common_mask = torch.zeros(self.num_writers, dtype=self._dtype)
        read_common_mask[:self.writer_idx] = 1
        self.register_buffer("read_common_mask", read_common_mask)
        
        write_common_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        write_common_mask[self.writer_idx, 0] = 1
        self.register_buffer("write_common_mask", write_common_mask)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic

    @torch.no_grad()
    def get_edge_masks(self):
        z = get_mask(self.mlp_read_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        z = z[:self.writer_idx]
        return (z,)
    
    @torch.no_grad()
    def get_node_masks(self):
        z = get_mask(self.mlp_write_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        return (z,)

    def read(self, x, corr_x=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x is (writers, batch_size, sequence_length, hidden_size)
        m = get_mask(self.mlp_read_log_alphas, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        
        z = m * self.read_common_mask
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-m) * self.read_common_mask)
        
        x_z = self.input_layernorm(x_z)
        
        z_edges_sum = torch.sum(z)
        
        return x_z, z_edges_sum

    def write(self, residual, x, corr_x=None):
        # x is (batch_size, sequence_length, hidden_size)
        # corr_x is (writers, batch_size, sequence_length, hidden_size)
        # residual is (writers, batch_size, sequence_length, hidden_size)
        z = get_mask(self.mlp_write_log_alphas, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        x = x * z
        
        if corr_x is not None:
            x = x + corr_x[self.writer_idx] * (1 - z)
        
        x = torch.einsum("ibsd,wi->wbsd", x.unsqueeze(0), self.write_common_mask)
        residual = residual + x
        
        return residual, z

    def load_proj(self, up_proj, down_proj, up_bias=None, down_bias=None):
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.up_bias = up_bias
        self.down_bias = down_bias
        self.initialized = True

    def forward(self, hidden_states, corr_x=None):
        residual = hidden_states
        
        hidden_states, z_edges_sum = self.read(hidden_states, corr_x=corr_x)
        
        hidden_states = torch.einsum("bsh,hk->bsk", hidden_states, self.up_proj)
        if self.up_bias is not None:
            hidden_states = hidden_states + self.up_bias.reshape(1, 1, -1)
        hidden_states = self.act(hidden_states)
        hidden_states = torch.einsum("bsk,kh->bsh", hidden_states, self.down_proj)
        if self.down_bias is not None:
            hidden_states = hidden_states + self.down_bias.reshape(1, 1, -1)
        
        residual, z_nodes_sum = self.write(residual, hidden_states, corr_x=corr_x)
        
        return residual, z_edges_sum, z_nodes_sum

class ErazrBlock(nn.Module):
    def __init__(
        self, 
        config: ErazrConfig,
        block_idx: int,
    ):
        super().__init__()
        
        self.block_idx = block_idx
        
        self.attn = ErazrAttention(
            config,
            reader_idx=block_idx * (1 + 3 * config.num_heads),
            writer_idx=(2 + block_idx * (1 + config.num_heads)),
        )
        self.mlp = ErazrMLP(
            config,
            reader_idx=((block_idx+1) * (1 + 3 *config.num_heads) - 1),
            writer_idx=(2 + (block_idx+1) * (1 + config.num_heads) - 1),
        )

    def load_weights(self, w_q, w_k, w_v, w_o, up_proj, down_proj, b_q=None, b_k=None, b_v=None, b_o=None, b_up=None, b_down=None):
        self.attn.load_w_qkv(w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o)
        self.mlp.load_proj(up_proj, down_proj, b_up, b_down)

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.attn.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)
        self.mlp.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)

    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.attn.set_node_threshold_for_deterministic(node_threshold_for_deterministic)
        self.mlp.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        attn_masks = self.attn.get_edge_masks()
        mlp_masks = self.mlp.get_edge_masks()
        return attn_masks + mlp_masks
    
    @torch.no_grad()
    def get_node_masks(self):
        attn_masks = self.attn.get_node_masks()
        mlp_masks = self.mlp.get_node_masks()
        return attn_masks + mlp_masks

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        corr_x=None,
    ):
        # hidden_states is noe (writers, batch_size, sequence_length, hidden_size)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            corr_x=corr_x,
        )
        hidden_states, z_attn_edges_sum, z_attn_nodes_sum = attn_outputs[0], attn_outputs[1], attn_outputs[2]
        
        hidden_states, z_mlp_edges_sum, z_mlp_nodes_sum = self.mlp(hidden_states, corr_x=corr_x)
        
        z_edges_sum = z_attn_edges_sum + z_mlp_edges_sum
        z_nodes_sum = z_attn_nodes_sum + z_mlp_nodes_sum

        if use_cache:
            outputs = (hidden_states, z_edges_sum, z_nodes_sum,) + attn_outputs
        else:
            outputs = (hidden_states, z_edges_sum, z_nodes_sum,) + attn_outputs[1:]

        return outputs  # hidden_states, z_edges_sum, z_nodes_sum, (attentions, cross_attentions)


class ErazrPretrainedModel(PreTrainedModel):
    config_class = ErazrConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["ErazrBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

@dataclass
class ErazrOutput:
    last_hidden_state: torch.FloatTensor
    past_key_values: Tuple[Tuple[torch.FloatTensor]]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    writer_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None

class ErazrModel(ErazrPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model_size = config.model_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.causal = config.causal
        self.num_total_edges = n_total_edges(config)
        self.num_writers = n_writers(config)
        self.num_readers = n_readers(config)
        self._dtype = dtype_to_torch[config.dtype]
        
        self.embedding = nn.Embedding(config.vocab_size, config.model_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.model_size)
        
        self.blocks = nn.ModuleList(
            [
                ErazrBlock(config, block_idx=l) 
                for l in range(config.num_layers)
            ]
        )
        
        self.initialized = False
        
        self.final_read_log_alpha = nn.Parameter(torch.empty(self.num_writers, dtype=self._dtype))
        self.final_read_log_alpha.data.normal_(mean=10.0, std=0.01)
        
        self.token_write_log_alpha = nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.pos_write_log_alpha = nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.token_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        self.pos_write_log_alpha.data.normal_(mean=10.0, std=0.01)
        
        token_write_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        token_write_mask[0, 0] = 1
        pos_write_mask = torch.zeros((self.num_writers, 1), dtype=self._dtype)
        pos_write_mask[1, 0] = 1
        self.register_buffer("token_write_mask", token_write_mask)
        self.register_buffer("pos_write_mask", pos_write_mask)
        
        self.sparsity_lambda_edges_1 = torch.nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.sparsity_lambda_edges_2 = torch.nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.sparsity_lambda_nodes_1 = torch.nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        self.sparsity_lambda_nodes_2 = torch.nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        
        self.edge_threshold_for_deterministic = None
        self.node_threshold_for_deterministic = None
        
        self.model_parallel = False
    
    @torch.no_grad()
    def load_everything(self, embeds_weights, pos_embeds_weights, block_weights):
        self.embedding.weight.data = embeds_weights
        self.position_embedding.weight.data = pos_embeds_weights
        for block, weights in zip(self.blocks, block_weights):
            block.load_weights(*weights)
        self.initialized = True

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.edge_threshold_for_deterministic = edge_threshold_for_deterministic
        for block in self.blocks:
            block.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)
            
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.node_threshold_for_deterministic = node_threshold_for_deterministic
        for block in self.blocks:
            block.set_node_threshold_for_deterministic(node_threshold_for_deterministic)                

    @torch.no_grad()
    def get_edge_masks(self):
        masks = []
        for block in self.blocks:
            masks.append(block.get_edge_masks())
        z = get_mask(self.final_read_log_alpha, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        masks.append((z,))
        return masks

    @torch.no_grad()
    def get_node_masks(self):
        z_token = get_mask(self.token_write_log_alpha, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_pos = get_mask(self.pos_write_log_alpha, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        masks = [(z_token, z_pos,)]
        
        for block in self.blocks:
            masks.append(block.get_node_masks())
        return masks
    
    @torch.no_grad()
    def get_edges(self, ignore_nodes=False):
        edge_masks = self.get_edge_masks()
        node_masks = self.get_node_masks()
        
        allowed_writers = []
        edges = []
        if node_masks[0][0] == 1:
            allowed_writers.append(0)
        if node_masks[0][1] == 1:
            allowed_writers.append(1)
        for l in range(self.num_layers):
            attn_writers = node_masks[l+1][0]
            for i in range(self.num_heads):
                if attn_writers[i] == 1:
                    allowed_writers.append(2 + l * (1 + self.num_heads) + i)
            mlp_writers = node_masks[l+1][1]
            if mlp_writers == 1:
                allowed_writers.append(2 + (l+1) * (1 + self.num_heads) - 1)
        
            attn_q_edges, attn_k_edges, attn_v_edges, mlp_edges = edge_masks[l]
            for from_idx in range(attn_q_edges.shape[0]):
                if (from_idx not in allowed_writers) and (not ignore_nodes):
                    continue
                for head_no in range(attn_q_edges.shape[1]):
                    if attn_q_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + 3 * self.num_heads) + head_no
                        edges.append((get_writer_name(from_idx, self.num_heads), get_reader_name(to_idx, self.num_heads)))
                for head_no in range(attn_k_edges.shape[1]):
                    if attn_k_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + 3 * self.num_heads) + self.num_heads + head_no
                        edges.append((get_writer_name(from_idx, self.num_heads), get_reader_name(to_idx, self.num_heads)))
                for head_no in range(attn_v_edges.shape[1]):
                    if attn_v_edges[from_idx, head_no] == 1:
                        to_idx = l * (1 + 3 * self.num_heads) + 2 * self.num_heads + head_no
                        edges.append((get_writer_name(from_idx, self.num_heads), get_reader_name(to_idx, self.num_heads)))
            for from_idx in range(mlp_edges.shape[0]):
                if (from_idx not in allowed_writers) and (not ignore_nodes):
                    continue
                if mlp_edges[from_idx] == 1:
                    to_idx = (l+1) * (1 + 3 * self.num_heads) - 1
                    edges.append((get_writer_name(from_idx, self.num_heads), get_reader_name(to_idx, self.num_heads)))
        final_read_mask = edge_masks[self.num_layers][0]
        for from_idx in range(self.num_writers):
            if (from_idx in allowed_writers or ignore_nodes) and (final_read_mask[from_idx] == 1):
                edges.append((get_writer_name(from_idx, self.num_heads), f"resid_post"))
        return edges

    def read(self, x, corr_x=None):
        # x is (writers, batch_size, sequence_length, hidden_size)
        # corr_x is (writers, batch_size, sequence_length, hidden_size)
        # embeds are (batch_size, sequence_length, hidden_size) and always contribute
        z = get_mask(self.final_read_log_alpha, training=self.training, threshold_for_deterministic=self.edge_threshold_for_deterministic)
        x_z = torch.einsum("wbsd,w->bsd", x, z)
        if corr_x is not None:
            x_z = x_z + torch.einsum("wbsd,w->bsd", corr_x, (1-z))
        
        z_sum = torch.sum(z)
        
        return x_z, z_sum

    def write(self, tok_embeds, pos_embeds, corr_x=None):
        z_tok = get_mask(self.token_write_log_alpha, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        z_pos = get_mask(self.pos_write_log_alpha, training=self.training, threshold_for_deterministic=self.node_threshold_for_deterministic)
        
        z_nodes_sum = z_tok + z_pos
        
        tok_embeds = tok_embeds * z_tok
        if corr_x is not None:
            tok_embeds = tok_embeds + corr_x[0] * (1 - z_tok)
        
        pos_embeds = pos_embeds * z_pos
        if corr_x is not None:
            pos_embeds = pos_embeds + corr_x[1] * (1 - z_pos)
            
        residual = torch.einsum("bsdh,wh->wbsd", tok_embeds.unsqueeze(-1), self.token_write_mask) + torch.einsum("bsdh,wh->wbsd", pos_embeds.unsqueeze(-1), self.pos_write_mask)
        return residual, z_nodes_sum

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        output_writer_states=False,
        return_dict=False,
        target_edge_sparsity=None,
        target_node_sparsity=None,
        corr_x=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.blocks))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, input_shape[-1], input_shape[-1], device=device)
        else:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        
        if self.causal:
            attention_mask = torch.tril(attention_mask)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states, z_nodes_sum = self.write(inputs_embeds, position_embeds, corr_x=corr_x)

        output_shape = hidden_states.shape

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = (embeds,) if output_hidden_states else None
        
        z_edges_sum = 0
        
        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                corr_x=corr_x,
            )

            hidden_states, z_block_edges_sum, z_block_nodes_sum = outputs[0], outputs[1], outputs[2]
            z_edges_sum = z_edges_sum + z_block_edges_sum
            z_nodes_sum = z_nodes_sum + z_block_nodes_sum
            
            if use_cache is True:
                presents = presents + (outputs[3],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[4 if use_cache else 3],)

        writer_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        hidden_states, z_final_edges_sum = self.read(writer_states, corr_x=corr_x)
        z_edges_sum = z_edges_sum + z_final_edges_sum
        
        model_edge_sparsity = 1 - (z_edges_sum / self.num_total_edges)
        model_node_sparsity = 1 - (z_nodes_sum / (self.num_writers))
        
        if target_edge_sparsity is None:
            edge_loss = None
        else:
            edge_loss = self.sparsity_lambda_edges_2 * (model_edge_sparsity - target_edge_sparsity) ** 2  + self.sparsity_lambda_edges_1 * (model_edge_sparsity - target_edge_sparsity)          

        if target_node_sparsity is None:
            node_loss = None
        else:
            node_loss = self.sparsity_lambda_nodes_2 * (model_node_sparsity - target_node_sparsity) ** 2 + self.sparsity_lambda_nodes_1 * (model_node_sparsity - target_node_sparsity)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states, None if not output_writer_states else writer_states,
                    presents, all_hidden_states, all_self_attentions, 
                    None if target_edge_sparsity is None else torch.tensor(target_edge_sparsity), 
                    None if target_node_sparsity is None else torch.tensor(target_node_sparsity), 
                    model_edge_sparsity, model_node_sparsity, edge_loss, node_loss
                ]
                if v is not None
            )

        return ErazrOutput(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            writer_states=None if not output_writer_states else writer_states, 
            attentions=all_self_attentions,
            target_edge_sparsity=None if target_edge_sparsity is None else torch.tensor(target_edge_sparsity),
            target_node_sparsity=None if target_node_sparsity is None else torch.tensor(target_node_sparsity),
            model_edge_sparsity=model_edge_sparsity,
            model_node_sparsity=model_node_sparsity,
            edge_loss=edge_loss,
            node_loss=node_loss,
        )


@dataclass
class ErazrForSequenceTransformationOutput:
    logits: torch.FloatTensor
    last_hidden_state: torch.FloatTensor
    past_key_values: Tuple[Tuple[torch.FloatTensor]]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    writer_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None

class ErazrModelForSequenceTransformation(ErazrPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.unembedding = None
        self.Erazr_model = ErazrModel(config)        
        self.initialized = False
        
        self.model_parallel = False
    
    @torch.no_grad()
    def load_everything(self, embeds_weights, pos_embeds_weights, unembedding_mtx, block_weights):
        self.unembedding = unembedding_mtx
        self.Erazr_model.load_everything(embeds_weights, pos_embeds_weights, block_weights)
        self.initialized = True

    @torch.no_grad()
    def set_edge_threshold_for_deterministic(self, edge_threshold_for_deterministic):
        self.Erazr_model.set_edge_threshold_for_deterministic(edge_threshold_for_deterministic)
            
    @torch.no_grad()
    def set_node_threshold_for_deterministic(self, node_threshold_for_deterministic):
        self.Erazr_model.set_node_threshold_for_deterministic(node_threshold_for_deterministic)

    @torch.no_grad()
    def get_edge_masks(self):
        return self.Erazr_model.get_edge_masks()
    
    @torch.no_grad()
    def get_node_masks(self):
        return self.Erazr_model.get_node_masks()
    
    @torch.no_grad()
    def get_edges(self):
        return self.Erazr_model.get_edges()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        output_writer_states=False,
        return_dict=False,
        target_edge_sparsity=None,
        target_node_sparsity=None,
        corr_x=None,
        *args,
        **kwargs,
    ):
        outputs = self.Erazr_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_writer_states=output_writer_states,
            return_dict=return_dict,
            target_edge_sparsity=target_edge_sparsity,
            target_node_sparsity=target_node_sparsity,
            corr_x=corr_x,
        )
        
        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        projected = torch.einsum("...sd,dv->...sv", hidden_states, self.unembedding)
        
        if not return_dict:
            if output_hidden_states:
                return (projected,) + outputs[1:]
            else:
                return (projected,) + outputs

        return ErazrForSequenceTransformationOutput(
            logits=projected,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=None if not output_hidden_states else outputs.hidden_states,
            writer_states=outputs.writer_states,
            attentions=outputs.attentions,
            target_edge_sparsity=outputs.target_edge_sparsity,
            target_node_sparsity=outputs.target_node_sparsity,
            model_edge_sparsity=outputs.model_edge_sparsity,
            model_node_sparsity=outputs.model_node_sparsity,
            edge_loss=outputs.edge_loss,
            node_loss=outputs.node_loss,
        )
        

if __name__ == '__main__':
    np.set_printoptions(threshold=100000)

    in_path = "data/tracr_models/xproportion.tracr.pkl"
    config, tok_embed, pos_embed, blocks_embeds, vocab, bos, pad, _ = get_config_weights_and_vocab(in_path)

    tokenizer = ErazrTokenizer(vocab, bos, pad)
    model = ErazrModel(config).to("cuda")
    model.set_edge_threshold_for_deterministic(0.00001)
    model.set_node_threshold_for_deterministic(0.00001)
    model.load_everything(tok_embed, pos_embed, blocks_embeds)

    example_encoded = tokenizer.encode([["BOS", "x", "a", "c", "x"]], return_tensors="pt").to("cuda")
    outputs = model(example_encoded, return_dict=True)
    
    print(outputs.last_hidden_state[0, :, 0])
    print(outputs.last_hidden_state.shape)
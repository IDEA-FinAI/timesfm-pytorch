import logging
import math
import multiprocessing
from os import path
import time
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from utilsforecast.processing import make_future_dataframe

PAD_VAL = 1123581321.0
DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_TOLERANCE = 1e-7

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropout_prob=0.0, layer_norm=False):
        super().__init__()
        self.layer_norm = layer_norm
        
        # self.hidden_layer = nn.Sequential(
        #     nn.Linear(input_dims, hidden_dims),
        #     nn.SiLU()  # Swish activation
        # )
        self.hidden_layer = nn.Linear(input_dims, hidden_dims)
        
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.dropout = nn.Dropout(dropout_prob)
        self.residual_layer = nn.Linear(input_dims, output_dims)
        
        if layer_norm:
            self.ln_layer = nn.LayerNorm(output_dims)

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.output_layer(hidden)
        output = self.dropout(output)
        residual = self.residual_layer(inputs)
        
        if self.layer_norm:
            return self.ln_layer(output + residual)
        else:
            return output + residual

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dims, num_heads=16, head_dim=80):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query = nn.Linear(input_dims, num_heads * head_dim, bias=True)
        self.key = nn.Linear(input_dims, num_heads * head_dim, bias=True)
        self.value = nn.Linear(input_dims, num_heads * head_dim, bias=True)
        self.post = nn.Linear(num_heads * head_dim, input_dims, bias=True)
        self.per_dim_scale = nn.Parameter(torch.ones(head_dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out * self.per_dim_scale
        out = out.view(batch_size, seq_len, -1)
        out = self.post(out)
        return out
    

class SelfAttention(nn.Module):
    def __init__(self, model_dims, num_heads=16, head_dim=80):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query = nn.Linear(model_dims, num_heads * head_dim, bias=True)
        self.key = nn.Linear(model_dims, num_heads * head_dim, bias=True)
        self.value = nn.Linear(model_dims, num_heads * head_dim, bias=True)
        self.post = nn.Linear(num_heads * head_dim, model_dims, bias=True)
        self.per_dim_scale = nn.Parameter(torch.ones(head_dim))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        # Scale by per-dimension scale
        out = out * self.per_dim_scale
        out = out.view(batch_size, seq_len, -1)
        out = self.post(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dims, bias=False)
        self.self_attention = SelfAttention(model_dims)
        self.ff_layer = nn.Sequential(
            nn.Linear(model_dims, model_dims, bias=True),
            nn.ReLU(),
            nn.Linear(model_dims, model_dims, bias=True)
        )
        self.ff_layer_norm = nn.LayerNorm(model_dims, bias=True)
        
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm(x))
        x = x + self.ff_layer(self.ff_layer_norm(x))
        return x

# class StackedTransformer(nn.Module):
#     def __init__(self, model_dims, hidden_dims, num_layers=6, nhead=8):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=model_dims, nhead=nhead, dim_feedforward=hidden_dims, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, src, src_key_padding_mask):
#         return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dims, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embedding_dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2).float() * (-math.log(10000.0) / embedding_dims))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_length):
        return self.pe[:, :seq_length]

class PatchedTimeSeriesDecoder(nn.Module):
    def __init__(self, 
                 patch_len, 
                 horizon_len, 
                 model_dims, 
                 hidden_dims, 
                 num_layers=6,
                 nhead=8,
                 quantiles=None,
                 use_freq=True):
        super().__init__()
        self.patch_len = patch_len
        self.horizon_len = horizon_len
        self.model_dims = model_dims
        self.hidden_dims = hidden_dims
        self.quantiles = quantiles if quantiles is not None else DEFAULT_QUANTILES
        self.use_freq = use_freq

        num_outputs = len(self.quantiles) + 1

        # self.stacked_transformer_layer = StackedTransformer(model_dims, hidden_dims, num_layers=num_layers, nhead=nhead)
        self.stacked_transformer_layer = nn.ModuleList([TransformerLayer(model_dims) for _ in range(num_layers)])
        
        self.input_ff_layer = ResidualBlock(2 * patch_len, hidden_dims, model_dims)
        print(f"patch_len: {patch_len}, hidden_dims: {hidden_dims}, model_dims: {model_dims}")
        self.horizon_ff_layer = ResidualBlock(model_dims, hidden_dims, horizon_len * num_outputs)
        
        # self.position_emb = PositionalEmbedding(model_dims)
        
        if self.use_freq:
            self.freq_emb = nn.Embedding(3, model_dims)

    def _masked_mean_std(self, inputs, padding):
        mask = 1 - padding
        num_valid_elements = torch.sum(mask, dim=2)
        num_valid_elements = torch.where(num_valid_elements == 0, torch.ones_like(num_valid_elements), num_valid_elements)
        
        masked_sum = torch.sum(inputs * mask, dim=2)
        masked_squared_sum = torch.sum((inputs * mask) ** 2, dim=2)
        
        masked_mean = masked_sum / num_valid_elements
        masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
        masked_var = torch.clamp(masked_var, min=0.0)
        masked_std = torch.sqrt(masked_var)
        
        return masked_mean, masked_std

    def _forward_transform(self, inputs, patched_pads):
        mu, sigma = self._masked_mean_std(inputs, patched_pads)
        sigma = torch.where(sigma < _TOLERANCE, torch.ones_like(sigma), sigma)
        outputs = (inputs - mu.unsqueeze(1).unsqueeze(2)) / sigma.unsqueeze(1).unsqueeze(2)
        outputs = torch.where(torch.abs(inputs - PAD_VAL) < _TOLERANCE, PAD_VAL * torch.ones_like(outputs), outputs)
        return outputs, (mu, sigma)

    def _reverse_transform(self, outputs, stats):
        mu, sigma = stats
        return outputs * sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3) + mu.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    def _shift_padded_seq(self, mask, seq):
        num = seq.shape[1]
        first_zero_idx = (mask == 0).long().argmax(dim=1)
        idx_range = torch.arange(num, device=seq.device)
        shifted_idx = (idx_range.unsqueeze(0) - first_zero_idx.unsqueeze(1)) % num
        return torch.gather(seq, 1, shifted_idx.unsqueeze(2).expand(-1, -1, seq.size(2)))

    def _preprocess_input(self, input_ts, input_padding, pos_emb=None):
        batch_size, seq_len = input_ts.shape
        patched_inputs = input_ts.view(batch_size, -1, self.patch_len)
        input_padding = torch.where(torch.abs(input_ts - PAD_VAL) < _TOLERANCE, torch.ones_like(input_padding), input_padding)
        patched_pads = input_padding.view(batch_size, -1, self.patch_len)
        
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)
        
        patched_padding = torch.min(patched_pads, dim=-1)[0]

        if pos_emb is None:
            position_emb = self.position_emb(model_input.shape[1])
        else:
            position_emb = pos_emb
        
        if not self.training:
            if position_emb.shape[0] != model_input.shape[0]:
                position_emb = position_emb.repeat(model_input.shape[0], 1, 1)
            position_emb = self._shift_padded_seq(patched_padding, position_emb)
        
        model_input += position_emb

        return model_input, patched_padding, stats, patched_inputs

    def _postprocess_output(self, model_output, num_outputs, stats):
        output_ts = self.horizon_ff_layer(model_output)
        output_ts = output_ts.view(output_ts.shape[0], output_ts.shape[1], self.horizon_len, num_outputs)
        return self._reverse_transform(output_ts, stats)

    def forward(self, inputs):
        input_ts, input_padding = inputs['input_ts'], inputs['input_padding']
        num_outputs = len(self.quantiles) + 1
        
        model_input, patched_padding, stats, _ = self._preprocess_input(input_ts, input_padding)
        
        if self.use_freq:
            freq = inputs['freq'].long()
            f_emb = self.freq_emb(freq)  # B x 1 x D
            f_emb = f_emb.repeat(1, model_input.shape[1], 1)
            model_input += f_emb
        
        model_output = self.stacked_transformer_layer(model_input, src_key_padding_mask=patched_padding)
        
        output_ts = self._postprocess_output(model_output, num_outputs, stats)
        
        return {
            'output_tokens': model_output,
            'output_ts': output_ts,
            'stats': stats
        }

    def decode(self, inputs, horizon_len, output_patch_len=None, max_len=512):
        final_out = inputs['input_ts']
        inp_time_len = final_out.shape[1]
        paddings = inputs['input_padding']
        if self.use_freq:
            freq = inputs['freq'].long()
        else:
            freq = torch.zeros(final_out.shape[0], 1, dtype=torch.long, device=final_out.device)
        
        full_outputs = []
        
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(f"Length of paddings must match length of input + horizon_len: {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")
        
        if output_patch_len is None:
            output_patch_len = self.horizon_len
        
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
        
        for _ in range(num_decode_patches):
            current_padding = paddings[:, :final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]
            model_input = {
                'input_ts': input_ts,
                'input_padding': input_padding,
                'freq': freq,
            }
            fprop_outputs = self(model_input)['output_ts']
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            full_outputs.append(fprop_outputs[:, -1, :output_patch_len, :])
            final_out = torch.cat([final_out, new_ts], dim=-1)

        return (
            final_out[:, inp_time_len:inp_time_len + horizon_len],
            torch.cat(full_outputs, dim=1)[:, :horizon_len, :]
        )
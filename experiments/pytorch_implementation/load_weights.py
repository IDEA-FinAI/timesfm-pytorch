import timesfm
import timesfm_pytorch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax
import torch

def recursive_convert(pytree, prefix=''):
    torch_state = {}
    # 只保存pytree结构，不保存数据
    for key, value in pytree.items():
        # print(key, value.keys() if isinstance(value, dict) else None)
        if isinstance(value, dict):
            torch_state.update(recursive_convert(value, f'{prefix}{key}.'))
        elif isinstance(value, (np.ndarray, jax.Array)):
            torch_state[f'{prefix}{key}'] = torch.tensor(np.array(value))
        else:
            print(f"Unsupported parameter type: {type(value)}")
            raise ValueError(f"Unsupported parameter type: {type(value)}")
    return torch_state

def convert_to_new_state_dict(state_dict):
    # 创建一个新的state_dict来存储重命名后的权重
    new_state_dict = {}

    for k, v in state_dict.items():
        if 'stacked_transformer_layer.x_layers_' in k:
            # 重命名transformer层的权重
            new_k = k.replace('stacked_transformer_layer.x_layers_', 'stacked_transformer_layer.')
            new_k = new_k.replace('.linear.w', '.weight')
            new_k = new_k.replace('.bias.b', '.bias')
            new_k = new_k.replace('ff_layer.ffn_layer1', 'ff_layer.0')
            new_k = new_k.replace('ff_layer.ffn_layer2', 'ff_layer.2')
            new_k = new_k.replace('ff_layer.layer_norm', 'ff_layer_norm')
            new_k = new_k.replace('.scale', '.weight')
            new_k = new_k.replace('self_attention.per_dim_scale.per_dim_scale', 'self_attention.per_dim_scale')
            
            # 修复self-attention部分的命名
            if 'self_attention' in new_k:
                new_k = new_k.replace('.w', '.weight')
                new_k = new_k.replace('.b', '.bias')
        elif 'freq_emb.emb_var' in k:
            new_k = 'freq_emb.weight'
        elif 'ff_layer' in k:
            # 处理FeedForwardLayer的权重
            new_k = k.replace('.linear.w', '.weight')
            new_k = new_k.replace('.bias.b', '.bias')
        else:
            # 重命名其他层的权重
            new_k = k.replace('.linear.w', '.weight')
            new_k = k.replace('.bias.b', '.bias')
        
        # 合并多头注意力,torch.Size([1280, 16, 80])合并成torch.Size([1280, 1280])
        if any(item in new_k for item in ['self_attention.query.weight',"self_attention.key.weight","self_attention.value.weight", "self_attention.post.weight"]):
            new_v = v.reshape(1280, 1280).t()
        # torch.Size([16, 80])合并为torch.Size([1280])
        elif any(item in new_k for item in ['self_attention.query.bias',"self_attention.key.bias","self_attention.value.bias"]):
            new_v = v.reshape(1280).t()
        # linear层
        elif any(item in new_k for item in ['input_ff_layer.hidden_layer.weight', 'input_ff_layer.residual_layer.weight']):
            new_v = v.t()
        else:
            new_v = v

        new_state_dict[new_k] = new_v

    # 处理output_layer的权重
    if 'output_layer.weight' not in new_state_dict and 'output_layer.linear.w' in new_state_dict:
        new_state_dict['output_layer.weight'] = new_state_dict.pop('output_layer.linear.w')
    if 'output_layer.bias' not in new_state_dict and 'output_layer.bias.b' in new_state_dict:
        new_state_dict['output_layer.bias'] = new_state_dict.pop('output_layer.bias.b')

    return new_state_dict



if __name__ == '__main__':

    context_len = 256
    horizon_len = 10

    tfm = timesfm.TimesFm(
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend='gpu',
    )

    tfm.load_from_checkpoint(checkpoint_path="/finance_ML/wuxiaojun/pretrained/Time/timesfm-1.0-200m/checkpoints")

    pytree = tfm._train_state.mdl_vars['params']
    torch_state = recursive_convert(pytree)

    # 初始化模型
    tfm_pytorch = timesfm_pytorch.TimesFm(
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=32,
        output_patch_len=128,
        model_dims=1280,
        num_layers=20,
        nhead=16
    )

    new_state_dict = convert_to_new_state_dict(state_dict=torch_state)
    tfm_pytorch.load_from_torch_state(torch_state=new_state_dict)

    print("Model loaded successfully.")

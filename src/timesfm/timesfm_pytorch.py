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

from patched_decoder_pytorch import PatchedTimeSeriesDecoder

PAD_VAL = 1123581321.0
DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_TOLERANCE = 1e-7
def freq_map(freq: str):
    freq = str.upper(freq)
    if (freq.endswith("H") or freq.endswith("T") or freq.endswith("MIN") or 
        freq.endswith("D") or freq.endswith("B") or freq.endswith("U")):
        return 0
    elif freq.endswith(("W", "M", "MS")):
        return 1
    elif freq.endswith("Y") or freq.endswith("Q"):
        return 2
    else:
        raise ValueError(f"Invalid frequency: {freq}")

def moving_average(arr, window_size):
    arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
    smoothed_arr = (np.convolve(arr_padded, np.ones(window_size), "valid") / window_size)
    return [smoothed_arr, arr - smoothed_arr]

def process_group(key, group, value_name, forecast_context_len):
    group = group.tail(forecast_context_len)
    return np.array(group[value_name], dtype=np.float32), key

class TimesFm:
    def __init__(
        self,
        context_len: int,
        horizon_len: int,
        input_patch_len: int,
        output_patch_len: int,
        num_layers: int,
        nhead: int,
        model_dims: int,
        per_core_batch_size: int = 32,
        backend: Literal["cpu", "cuda"] = "cpu",
        quantiles: Sequence[float] | None = None,
        verbose: bool = True,
    ) -> None:
        self.per_core_batch_size = per_core_batch_size
        self.backend = backend
        self.num_devices = torch.cuda.device_count() if backend == "cuda" else 1
        self.global_batch_size = self.per_core_batch_size * self.num_devices

        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len

        if quantiles is None:
            quantiles = DEFAULT_QUANTILES

        self.model = PatchedTimeSeriesDecoder(
            patch_len=input_patch_len,
            horizon_len=output_patch_len,
            model_dims=model_dims,
            hidden_dims=model_dims,
            num_layers=num_layers,
            nhead=nhead,
            quantiles=quantiles,
            use_freq=True
        )

        self._verbose = verbose
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("Multiprocessing context has already been set.")

    def _logging(self, s):
        if self._verbose:
            print(s)

    def load_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        repo_id: str = "google/timesfm-1.0-200m",
        step: int | None = None,
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = path.join(snapshot_download(repo_id), "checkpoints")

        self._logging(f"Restoring checkpoint from {checkpoint_path}.")
        start_time = time.time()
        
        # Load the PyTorch state dict
        state_dict = torch.load(checkpoint_path, map_location=self.backend)
        self.model.load_state_dict(state_dict)
        
        self._logging(f"Restored checkpoint in {time.time() - start_time:.2f} seconds.")
        
        # Move model to appropriate device
        self.model = self.model.to(self.backend)
        self.model.eval()

    def load_from_torch_state(self, torch_state: dict):
        self.model.load_state_dict(torch_state)
        self.model = self.model.to(self.backend)
        self.model.eval()

    def _preprocess(self, inputs: Sequence[np.array], freq: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        input_ts, input_padding, inp_freq = [], [], []

        pmap_pad = ((len(inputs) - 1) // self.global_batch_size + 1) * self.global_batch_size - len(inputs)

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = np.concatenate([np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0)
                padding = np.concatenate([np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0)
            elif input_len > self.context_len:
                ts = ts[-self.context_len:]
                padding = padding[-(self.context_len + self.horizon_len):]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        for _ in range(pmap_pad):
            input_ts.append(input_ts[-1])
            input_padding.append(input_padding[-1])
            inp_freq.append(inp_freq[-1])
    

import hashlib
import json
import os
from typing import List, Tuple, Union

import torch

from common.interface import DirConfig


def load_dir_config(tmp_share_dir):
    with open(os.path.join(tmp_share_dir, "dir_config.json")) as f:
        dir_config = DirConfig(**json.load(f))
        assert dir_config.tmp_share_dir == tmp_share_dir
    return dir_config


def hash_tensor(*tensors: torch.Tensor):
    h = hashlib.sha256()
    cpu = torch.device("cpu")
    for t in tensors:
        if t.device != cpu:
            t = t.cpu()

        if t.dtype == torch.bfloat16:
            h.update(t.float().detach().numpy().tobytes())
        else:
            h.update(t.detach().numpy().tobytes())
    return h.digest()


def nextafter(x: torch.Tensor, offset: torch.Tensor):
    MAX_EXPONENT = 0xFF
    MAX_MANTISSA = 0x7FFFFF

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    if not torch.is_tensor(offset):
        offset = torch.tensor(offset, dtype=torch.int32)

    x, offset = torch.broadcast_tensors(torch.clone(x), torch.clone(offset))

    assert not torch.is_floating_point(offset)
    assert x.dtype == torch.float32
    assert (-MAX_MANTISSA <= offset).all() and (offset <= MAX_MANTISSA).all()

    x_int = x.view(torch.int32)

    # Extract the sign (1 bit), exponent (8 bits), and mantissa (23 bits)
    sign = (x_int >> 31) & 1
    exponent = (x_int >> 23) & MAX_EXPONENT
    mantissa = x_int & MAX_MANTISSA

    # If
    #   "sign is positive and offset is positive"
    # or
    #   "sign is negative and offset is negative"
    # treat the calculation as an addition that can overflow. Otherwise, it's
    # a substraction that can underflow.
    sign_is_positive = sign == 0
    offset = torch.where(sign_is_positive, offset, -offset)

    underflow = (mantissa + offset) < 0
    overflow = (mantissa + offset) > MAX_MANTISSA

    zero_pass = exponent == 0
    to_inf = exponent == MAX_EXPONENT

    # OVERFLOW
    # Handle "regular" overflow. We add the mantissa and the offset modulo the
    # max mantissa value and increase the exponent by one
    mantissa[overflow] = (mantissa[overflow] + offset[overflow]) & MAX_MANTISSA
    exponent[overflow & ~to_inf] += 1

    # If the exponent was already maximal, we set the mantissa to zero
    # (max exponent + non-zero mantissa is a NaN)
    # This is different from how torch.nextafter handles it. They return a NaN
    # value for some reason.
    mantissa[overflow & to_inf] = 0

    # UNDERFLOW
    # Handle "regular" underflow.
    underflow_r = underflow & ~zero_pass
    underflow_z = underflow & zero_pass

    mantissa[underflow_r] = (mantissa[underflow_r] + offset[underflow_r]) & MAX_MANTISSA
    exponent[underflow_r] -= 1

    # Underflow past zero
    mantissa[underflow_z] = -offset[underflow_z] - mantissa[underflow_z]
    sign[underflow_z] = 1 - sign[underflow_z]

    mantissa[~overflow & ~underflow] = (
        mantissa[~overflow & ~underflow] + offset[~overflow & ~underflow]
    )

    sign = sign & 0x1
    exponent = exponent & MAX_EXPONENT
    mantissa = mantissa & MAX_MANTISSA

    return ((sign << 31) | (exponent << 23) | mantissa).view(torch.float32)

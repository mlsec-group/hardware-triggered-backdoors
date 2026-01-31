import copy
import heapq
import io
import logging
from typing import Dict, List, Optional
import zlib
import torch

from strategies.client.backdoor_heuristics.heuristic import Heuristic, HeuristicOutput


def average_models(
    dst_model: torch.nn.Module,
    top_parameters: List[List[torch.Tensor]],
    *,
    weights=None,
):
    assert len(top_parameters) > 0

    if weights is None:
        weights = [1] * len(top_parameters)
    total_weights = sum(weights)

    with torch.no_grad():
        for dst_parameter in dst_model.parameters():
            dst_parameter.zero_()

        for model_params, w in zip(top_parameters, weights):
            for dst_parameter, src_parameter in zip(
                dst_model.parameters(), model_params
            ):
                dst_parameter.add_(w * src_parameter)

        for dst_parameter in dst_model.parameters():
            dst_parameter.div_(total_weights)


def crossover_models(
    dst_model: torch.nn.Module, top_parameters: List[List[torch.Tensor]], *, seed=None
):
    assert len(top_parameters) > 0

    num_models = len(top_parameters)

    # Create a generator for reproducible randomness
    device = next(dst_model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    with torch.no_grad():
        for dst_parameter, src_parameters in zip(
            dst_model.parameters(), zip(*top_parameters)
        ):
            # Stack candidate parameters: shape (num_models, *param_shape)
            stacked = torch.stack(src_parameters)  # (num_models, ...)

            # Sample random model indices for each element
            choices = torch.randint(
                low=0,
                high=num_models,
                size=dst_parameter.shape,
                device=device,
                generator=generator,
            )

            # Flatten for indexing
            flat_choices = choices.view(-1)
            flat_params = stacked.view(num_models, -1)

            # Correct advanced indexing
            selected = flat_params[
                flat_choices, torch.arange(flat_params.size(1), device=device)
            ]

            # Reshape back to parameter shape
            dst_parameter.copy_(selected.view_as(dst_parameter))

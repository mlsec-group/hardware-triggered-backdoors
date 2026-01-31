import copy
import heapq
import io
import logging
from typing import Dict, List, Optional
import zlib
import torch

from strategies.client.backdoor_heuristics.heuristic import Heuristic, HeuristicOutput


def c_and_w(y_pred, target_labels):
    # Create a mask to ignore the target class logit in the max operation
    mask = torch.ones_like(y_pred, dtype=torch.bool)
    mask.scatter_(1, target_labels.unsqueeze(1), False)

    # Apply the mask to the predicted logits (set target class logits to -inf)
    masked_logits = y_pred.masked_fill(~mask, float("-inf"))

    # Find the maximum logit for each row where i != t (i.e., excluding the target class)
    max_non_target_logits = torch.max(masked_logits, dim=1)[0]

    # Get the logits corresponding to the target class for each row
    target_logits = y_pred.gather(1, target_labels.unsqueeze(1)).squeeze(1)

    # Compute the loss for each row: max((max_non_target_logits - target_logits), 0)
    losses = torch.clamp(max_non_target_logits - target_logits, min=0)

    # Average the losses over all rows
    return torch.mean(losses)


def regularization(original_parameters, parameters):
    loss = 0
    size = 0

    for orig_param, cur_param in zip(original_parameters, parameters):
        loss += torch.sum((orig_param - cur_param) ** 2)
        size += torch.numel(orig_param)

    return loss

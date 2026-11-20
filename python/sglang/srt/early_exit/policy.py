# SPDX-License-Identifier: Apache-2.0
"""Classifier action selection shared by CUDA capture and offline validation."""

import torch
from torch import nn


FIVE_CLASS = "five_class_shared_mlp_v1"
FOUR_WIDTH_BCE = "four_width_bce_mlp_v1"


def load_classifier(saved, hidden_size):
    checkpoint_format = saved["format"]
    if checkpoint_format not in (FIVE_CLASS, FOUR_WIDTH_BCE):
        raise ValueError(f"Unsupported early-exit checkpoint format: {checkpoint_format}")
    depth = saved.get("depth", 1)
    if depth not in (0, 1) or (depth == 0 and checkpoint_format != FOUR_WIDTH_BCE):
        raise ValueError("early exit supports depth0 BCE or depth1 shared heads")
    layers = []
    if depth == 1:
        layers.extend(
            [
                nn.Linear(hidden_size, saved["width"], dtype=torch.float32),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
        )
    layers.append(
        nn.Linear(
            saved["width"] if depth else hidden_size,
            4 if checkpoint_format == FOUR_WIDTH_BCE else 5,
            dtype=torch.float32,
        )
    )
    net = nn.Sequential(*layers).eval().requires_grad_(False)
    net.load_state_dict(
        {k.removeprefix("net."): v for k, v in saved["state_dict"].items()}
    )
    return net


def select_action(logits, actions, threshold, checkpoint_format):
    if checkpoint_format == FOUR_WIDTH_BCE:
        # Each output independently predicts safety at its own width. Safety
        # need not be monotone: select the longest individually eligible width.
        eligible = logits.sigmoid() >= threshold
        return torch.where(eligible, actions[None, 1:], 0).amax(-1)
    if checkpoint_format == FIVE_CLASS:
        confidence, index = logits.softmax(-1).max(-1)
        return torch.where(confidence >= threshold, actions[index], 0)
    raise ValueError(f"Unsupported early-exit checkpoint format: {checkpoint_format}")

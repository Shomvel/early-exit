# SPDX-License-Identifier: Apache-2.0
"""Classifier action selection shared by CUDA capture and offline validation."""

import torch


FIVE_CLASS = "five_class_shared_mlp_v1"
FOUR_WIDTH_BCE = "four_width_bce_mlp_v1"


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

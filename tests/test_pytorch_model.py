"""
Unit tests for PyTorch TabularDeepNet architecture.
"""

import torch
import pytest
from src.ml.architectures import TabularDeepNet, get_embedding_dim

def test_embedding_dim_heuristic():
    assert get_embedding_dim(2) >= 4
    assert get_embedding_dim(100) <= 50

def test_tabular_deep_net_forward_binary():
    batch_size = 16
    num_continuous = 5
    cat_cards = [4, 10, 2] # 3 categorical features

    model = TabularDeepNet(
        num_continuous=num_continuous,
        cat_cardinalities=cat_cards,
        output_dim=1,
        task_type="binary_classification",
        hidden_dims=[32, 16],
        use_residual=True
    )

    x_cont = torch.randn(batch_size, num_continuous)
    x_cat = torch.randint(0, 2, (batch_size, len(cat_cards)))

    out = model(x_cont, x_cat)
    assert out.shape == (batch_size,)

def test_tabular_deep_net_only_continuous():
    batch_size = 8
    num_continuous = 4

    model = TabularDeepNet(
        num_continuous=num_continuous,
        cat_cardinalities=[],
        output_dim=1,
        task_type="regression"
    )

    x_cont = torch.randn(batch_size, num_continuous)
    out = model(x_cont, None)
    assert out.shape == (batch_size,)

def test_tabular_deep_net_single_sample_inference():
    model = TabularDeepNet(
        num_continuous=3,
        cat_cardinalities=[5, 2],
        output_dim=1,
        task_type="binary_classification"
    )
    x_cont = torch.randn(1, 3)
    x_cat = torch.tensor([[1, 0]], dtype=torch.long)
    out = model(x_cont, x_cat)
    assert out.ndim == 1
    prob = float(torch.sigmoid(out).view(-1)[0].item())
    assert 0.0 <= prob <= 1.0

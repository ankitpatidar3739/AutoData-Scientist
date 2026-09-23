"""
Custom Deep Learning PyTorch Architectures for Tabular Data.
Features:
- Dynamic Categorical Entity Embeddings (FastAI dimension heuristic)
- Continuous feature batch normalization and projection
- Residual / Skip connection Multi-Layer Perceptron (Tabular ResNet)
- Dropout, LayerNorm, and Mish/LeakyReLU activations
- Support for Binary Classification, Multiclass, and Regression
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_embedding_dim(cardinality: int) -> int:
    """FastAI rule of thumb for categorical embedding dimensions."""
    return min(50, max(4, int(round(1.6 * (cardinality ** 0.56)))))

class ResidualTabularBlock(nn.Module):
    """Residual block with LayerNorm, Linear, Mish activation, and Dropout."""
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.act(out + residual)

class TabularDeepNet(nn.Module):
    """
    Advanced Deep Learning Neural Network for Tabular Data.
    Combines learned entity embeddings for categorical features with normalized continuous features.
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        output_dim: int = 1,
        task_type: str = "binary_classification",
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_residual: bool = True
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities
        self.output_dim = output_dim
        self.task_type = task_type
        self.use_residual = use_residual

        # 1. Categorical Embedding Layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, get_embedding_dim(card))
            for card in cat_cardinalities
        ])
        total_emb_dim = sum(get_embedding_dim(card) for card in cat_cardinalities)

        # 2. Continuous Features Normalization
        if num_continuous > 0:
            self.cont_norm = nn.LayerNorm(num_continuous)
        else:
            self.cont_norm = None

        total_input_dim = total_emb_dim + num_continuous

        # 3. Dense & Residual Backbone
        hidden_dims = hidden_dims or [128, 64]
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

        layers = []
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        if self.use_residual:
            self.res_block = ResidualTabularBlock(current_dim, dropout_rate=dropout_rate)
        else:
            self.res_block = nn.Identity()

        # 4. Output Head
        self.head = nn.Linear(current_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for stability and faster convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x_cont: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tensors_to_concat = []

        # Process Categorical with defensive range clamping
        if self.embeddings and x_cat is not None and x_cat.shape[1] > 0:
            embedded = [emb(torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)) for i, emb in enumerate(self.embeddings)]
            cat_out = torch.cat(embedded, dim=1)
            tensors_to_concat.append(cat_out)

        # Process Continuous
        if self.cont_norm is not None and x_cont is not None and x_cont.shape[1] > 0:
            cont_out = self.cont_norm(x_cont)
            tensors_to_concat.append(cont_out)

        if not tensors_to_concat:
            raise ValueError("No input features provided to TabularDeepNet.")

        x = torch.cat(tensors_to_concat, dim=1)
        x = self.input_projection(x)
        x = self.backbone(x)
        x = self.res_block(x)
        logits = self.head(x)

        if self.output_dim == 1 and self.task_type in ("binary_classification", "regression"):
            logits = logits.squeeze(-1)

        return logits

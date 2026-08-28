"""The DTR two-tower model (Herzig et al., 2021).

    h_q = W_q . TAPAS_q(q)[CLS]
    h_T = W_T . TAPAS_T(title(T), T)[CLS]
    S(q, T) = h_q^T h_T

Kept faithful to tapas/models/table_retriever_model.py in google-research/tapas
on four points that are easy to get silently wrong:

- The two TAPAS towers are separate, with no weight sharing.
- [CLS] means the pooled output (dense + tanh), not the raw first hidden state.
- The projections are bias-free, shaped [proj_dim, hidden], and applied as
  matmul(h, W, transpose_b=True), which is exactly what nn.Linear computes.
- There is no L2 normalization and no logit scaling.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import TapasConfig, TapasModel

PROJECTION_DIM = 256


@dataclass
class DualEncoderOutput:
    query_embeddings: torch.Tensor  # (B, proj_dim)
    table_embeddings: torch.Tensor  # (B, proj_dim)
    logits: torch.Tensor  # (B, B), or (B, 2B) with mined negatives
    loss: torch.Tensor | None = None


class TapasDualEncoder(nn.Module):
    """Two TAPAS towers with down-projection to projection_dim."""

    def __init__(self, config: TapasConfig, projection_dim: int = PROJECTION_DIM):
        super().__init__()
        self.config = config
        self.projection_dim = projection_dim

        # Table tower first, mirroring the TF variable order where the table
        # encoder takes scope 'bert' and the query encoder 'bert_1'.
        self.table_encoder = TapasModel(config)
        self.query_encoder = TapasModel(config)

        self.table_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.query_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)

    def encode_tables(self, **inputs) -> torch.Tensor:
        return self.table_projection(self.table_encoder(**inputs).pooler_output)

    def encode_queries(self, **inputs) -> torch.Tensor:
        return self.query_projection(self.query_encoder(**inputs).pooler_output)

    def forward(self, query_inputs, table_inputs, negative_table_inputs=None):
        """Score a batch, treating every other table in it as a negative.

        Args:
            query_inputs: Tokenized queries, batch size B.
            table_inputs: Tokenized gold tables, aligned row-wise with the
                queries.
            negative_table_inputs: Optional mined hard negatives, one per query.
                When given, their scores are appended column-wise to produce a
                (B, 2B) logit matrix with label matrix [I | 0]. This is the
                original's row-wise concatenation of S and S'.

        Returns:
            A DualEncoderOutput carrying both towers' embeddings, the logits,
            and the softmax cross-entropy loss.
        """
        h_q = self.encode_queries(**query_inputs)
        h_t = self.encode_tables(**table_inputs)

        logits = h_q @ h_t.T
        if negative_table_inputs is not None:
            h_n = self.encode_tables(**negative_table_inputs)
            logits = torch.cat([logits, h_q @ h_n.T], dim=1)

        # Gold pairs sit on the diagonal, so the label for row i is just i.
        labels = torch.arange(h_q.size(0), device=h_q.device)
        loss = nn.functional.cross_entropy(logits, labels)

        return DualEncoderOutput(
            query_embeddings=h_q, table_embeddings=h_t, logits=logits, loss=loss
        )

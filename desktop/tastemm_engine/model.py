from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool as gmp, global_mean_pool as gap


class TasteBaselineModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_graph_features: int = 61,
        edge_attr_dim: int = 18,
        mixfp_dim: int = 3239,
        bert_dim: int = 768,
        gat_dim: int = 128,
        contrast_dim: int = 64,
        graph_aux_hidden_dim: int = 128,
        num_classes: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.mixfp_dim = mixfp_dim
        self.bert_dim = bert_dim

        self.gat1_heads = 8
        self.gat1 = GATv2Conv(
            num_graph_features,
            gat_dim,
            heads=self.gat1_heads,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )
        self.gat2 = GATv2Conv(
            gat_dim * self.gat1_heads,
            gat_dim,
            heads=1,
            concat=True,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )
        self.graph_res1 = nn.Linear(num_graph_features, gat_dim * self.gat1_heads)
        self.graph_res2 = nn.Linear(gat_dim * self.gat1_heads, gat_dim)
        self.graph_norm1 = nn.LayerNorm(gat_dim * self.gat1_heads)
        self.graph_norm2 = nn.LayerNorm(gat_dim)
        self.graph_proj = nn.Linear(gat_dim * 2, embed_dim)

        self.proj_mixfp = nn.Sequential(
            nn.Linear(mixfp_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.proj_bert = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self.fuse_feature = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        self.contrast_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, contrast_dim),
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.graph_aux_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, graph_aux_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_aux_hidden_dim, num_classes),
        )

    @staticmethod
    def _ensure_2d(feat: torch.Tensor, dim: int) -> torch.Tensor:
        if feat.dim() == 1:
            return feat.unsqueeze(0)
        if feat.dim() == 2 and feat.size(-1) == dim:
            return feat
        raise ValueError(f"Expected feature shape [B,{dim}] or [{dim}], got {tuple(feat.shape)}")

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 18), device=x.device, dtype=x.dtype)

        gx = self.gat1(x, edge_index, edge_attr=edge_attr)
        gx = F.elu(gx)
        gx = self.graph_norm1(gx + self.graph_res1(x))
        gx = F.dropout(gx, p=self.dropout, training=self.training)
        gx_res = gx

        gx = self.gat2(gx, edge_index, edge_attr=edge_attr)
        gx = F.elu(gx)
        gx = self.graph_norm2(gx + self.graph_res2(gx_res))

        graph_feat = torch.cat([gap(gx, batch), gmp(gx, batch)], dim=-1)
        graph_feat = self.graph_proj(graph_feat)
        graph_feat = F.relu(graph_feat)
        graph_feat = F.layer_norm(graph_feat, graph_feat.shape[1:])

        mixfp = self._ensure_2d(data.mixfp, self.mixfp_dim)
        mixfp_feat = self.proj_mixfp(mixfp)
        mixfp_feat = F.layer_norm(mixfp_feat, mixfp_feat.shape[1:])

        bert = self._ensure_2d(data.bert, self.bert_dim)
        bert_feat = self.proj_bert(bert)
        bert_feat = F.layer_norm(bert_feat, bert_feat.shape[1:])

        fused_feat = self.fuse_feature(torch.cat([graph_feat, mixfp_feat, bert_feat], dim=-1))
        return fused_feat, graph_feat

    def forward(self, data, mode: str = "classify"):
        fused_feat, graph_feat = self.encode(data)

        if mode == "contrastive":
            out = self.contrast_head(fused_feat)
            return F.normalize(out, p=2, dim=-1)
        if mode == "classify":
            return self.class_head(fused_feat)
        if mode == "features":
            return fused_feat
        if mode == "graph_features":
            return graph_feat
        if mode == "graph_aux":
            return self.graph_aux_head(graph_feat)
        if mode == "prob":
            return torch.softmax(self.class_head(fused_feat), dim=1)
        raise ValueError(f"Unsupported mode={mode!r}")

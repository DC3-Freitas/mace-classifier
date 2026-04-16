from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch_geometric
from loader import BRAVAIS_LABELS


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1, epsilon: float = 1e-4):
    # z = [batch_size, embed_dim]
    # L_var = 1/d sum (j=1)^d max(0, gamma - sqrt(Var(z_j) + epsilon))
    var = z.var(dim=0, unbiased=False)
    return torch.mean(torch.relu(gamma - torch.sqrt(var + epsilon)))


def vicreg_covariance_loss(z: torch.Tensor):
    # z = [batch_size, embed_dim]
    # L_cov = 1/(d(d-1)) sum (j!=k) ([C]_jk)^2
    cov = torch.cov(z.T)
    d = z.shape[1]
    off_diag_mask = ~torch.eye(d, dtype=bool, device=z.device)
    return torch.sum(cov[off_diag_mask] ** 2) / (d * (d - 1))


def ranking_loss(
    s: torch.Tensor,  # (num_graphs,) predicted order score
    pairs: list[
        Tuple[int, int]
    ],  # list of (idx_a, idx_b) where a should be less than b
    margin: float = 0.1,
):
    if len(pairs) == 0:
        return torch.tensor(0.0, device=s.device)
    loss = 0.0
    for idx_a, idx_b in pairs:
        loss += torch.relu(-(s[idx_a] - s[idx_b]) + margin)
    return loss / len(pairs)


def compute_pair_indices(batch: torch_geometric.data.Batch):
    """
    From batch metadata, compute: - chem_pairs: list of (graph_idx_a, graph_idx_b)
    where a,b share geometry but differ in species
    (for L_inv,S: structural head should be invariant)- struct_pairs: list of (graph_idx_a, graph_idx_b)
    where a,b share species but differ in geometry
    (for L_inv,C: chemical head should be invariant)- rank_S_pairs: list of (graph_idx_a, graph_idx_b)
    where sigma_a < sigma_b (same family, same chemistry)- rank_C_pairs: list of (graph_idx_a, graph_idx_b)
    where p_a < p_b (same family, same geometry)
    """
    chem_pairs = []
    struct_pairs = []
    rank_S_pairs = []
    rank_C_pairs = []
    for i in range(batch.num_graphs):
        for j in range(i + 1, batch.num_graphs):
            same_family = batch.family_id[i] == batch.family_id[j]
            same_chem = batch.bravais_label[i] == batch.bravais_label[j]
            same_struct = batch.ordering_type_label[i] == batch.ordering_type_label[j]
            if same_struct and not same_chem:
                chem_pairs.append((i, j))
            if same_chem and not same_struct:
                struct_pairs.append((i, j))
            if same_family and same_chem and same_struct:
                if batch.sigma[i] < batch.sigma[j]:
                    rank_S_pairs.append((i, j))
                elif batch.sigma[i] > batch.sigma[j]:
                    rank_S_pairs.append((j, i))
                if batch.shuffle_fraction[i] < batch.shuffle_fraction[j]:
                    rank_C_pairs.append((i, j))
                elif batch.shuffle_fraction[i] > batch.shuffle_fraction[j]:
                    rank_C_pairs.append((j, i))
    return {
        "chem_pairs": chem_pairs,
        "struct_pairs": struct_pairs,
        "rank_S_pairs": rank_S_pairs,
        "rank_C_pairs": rank_C_pairs,
    }


def batch_invariance_loss(embeddings, batch, pair_indices):
    """
    Compute invariance loss over all pairs in batch.
    pair_indices: list of (graph_idx_a, graph_idx_b)
    """
    total = 0.0
    for idx_a, idx_b in pair_indices:
        mask_a = batch.batch == idx_a
        mask_b = batch.batch == idx_b
        z_a = embeddings[mask_a]
        z_b = embeddings[mask_b]
        total += nn.MSELoss()(z_a.mean(dim=0), z_b.mean(dim=0))
    return total / max(len(pair_indices), 1)


def bravais_classification_loss(
    y_bravais: torch.Tensor, batch: torch_geometric.data.Batch
):
    # y_bravais: (total_num_graphs, num_classes)
    # batch.batch: (total_num_nodes,) with graph indices
    # batch.bravais_label: (total_num_graphs,) with class indices
    return nn.CrossEntropyLoss()(y_bravais, batch.bravais_label)


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        lambda_rank: float = 10.0,
        lambda_cls_S: float = 1.0,
        rank_margin: float = 0.1,
        vicreg_gamma: float = 1.0,
    ):
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_rank = lambda_rank
        self.lambda_cls_S = lambda_cls_S
        self.rank_margin = rank_margin
        self.vicreg_gamma = vicreg_gamma

        self.weights = {
            "var_S": lambda_var,
            "var_C": lambda_var,
            "cov_S": lambda_cov,
            "cov_C": lambda_cov,
            "rank_S": lambda_rank,
            "rank_C": lambda_rank,
            "cls_S": lambda_cls_S,
            "inv_S": lambda_inv,
            "inv_C": lambda_inv,
        }

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: torch_geometric.data.Batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: dict from model forward pass
            (z_S, z_C, s_hat, c_hat, y_bravais)
            batch: PyG Batch with metadata
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss values
        """
        pairs = compute_pair_indices(batch)
        loss_dict = {
            "cls_S": bravais_classification_loss(outputs["y_bravais"], batch),
            "var_S": vicreg_variance_loss(outputs["z_S"]),
            "var_C": vicreg_variance_loss(outputs["z_C"]),
            "cov_S": vicreg_covariance_loss(outputs["z_S"]),
            "cov_C": vicreg_covariance_loss(outputs["z_C"]),
            "rank_S": ranking_loss(outputs["s_hat"], pairs["rank_S_pairs"]),
            "rank_C": ranking_loss(outputs["c_hat"], pairs["rank_C_pairs"]),
            "inv_S": batch_invariance_loss(outputs["s_hat"], pairs["chem_pairs"]),
            "inv_C": batch_invariance_loss(outputs["c_hat"], pairs["struct_pairs"]),
        }
        total = sum(self.weights[k] * v for k, v in loss_dict.items())
        return total, loss_dict

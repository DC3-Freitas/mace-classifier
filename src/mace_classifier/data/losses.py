from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch_geometric

from mace_classifier.data.batchloader import MONO_SPECIES_SHUFFLE, PURE_PERMUTATION


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
    node_pairs: list[
        Tuple[torch.Tensor, torch.Tensor]
    ],  # for (a, b) things in a should be greater than that of b
    # Assumes same sizes
    margin: float = 0.1,
):
    if len(node_pairs) == 0:
        return torch.tensor(0.0, device=s.device)

    loss = 0.0

    for high_nodes, low_nodes in node_pairs:
        high_scores = s[high_nodes]
        low_scores = s[low_nodes]
        loss = loss + torch.relu(low_scores - high_scores + margin).mean()

    return loss / len(node_pairs)


# Alex: Combat collapse
def chem_pair_repulsion(
    z: torch.Tensor,
    node_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    margin: float = 1.0,
):
    if len(node_pairs) == 0:
        return torch.zeros((), device=z.device)

    loss = 0.0
    for nodes_a, nodes_b in node_pairs:
        diff = (z[nodes_a] - z[nodes_b]).norm(dim=-1)
        loss = loss + torch.relu(margin - diff).mean()

    return loss / len(node_pairs)


def compute_pair_indices(batch: torch_geometric.data.Batch):
    """
    From batch metadata, compute:
        - chem_pairs: list of (nodes a, nodes b)
          where a,b share geometry but differ in species
          (for L_inv,S: structural head should be invariant)
        - struct_pairs: list of (nodes a, nodes b)
          where a,b share species but differ in geometry
          (for L_inv,C: chemical head should be invariant)
        - rank_S_pairs: list of (nodes a, nodes b)
          where sigma_a < sigma_b (same family, same chemistry)
        - rank_C_pairs: list of (nodes a, nodes b)
          where p_a < p_b (same family, same geometry)
    """

    # Alex: made this node level and edited the corresponding things
    chem_pairs = []
    struct_pairs = []
    rank_S_pairs = []
    rank_C_pairs = []

    for i in range(batch.num_graphs):
        for j in range(i + 1, batch.num_graphs):
            same_family = batch.family_id[i] == batch.family_id[j]
            same_chem = (
                batch.bravais_label[i] == batch.bravais_label[j]
            ) and batch.shuffle_fraction[i] == batch.shuffle_fraction[j]
            same_struct = (
                batch.ordering_type_label[i] == batch.ordering_type_label[j]
            ) and batch.sigma[i] == batch.sigma[j]

            i_nodes = (batch.batch == i).nonzero(as_tuple=True)[0]
            j_nodes = (batch.batch == j).nonzero(as_tuple=True)[0]

            # Alex: Enforce family the same and bug fixes
            if same_family:
                # Fixed structure
                if same_struct and not same_chem:
                    chem_pairs.append((i_nodes, j_nodes))

                    # Alex: Don't rank monospecies or pure permutation since they introduce
                    # corner cases
                    if batch.shuffle_fraction[i] not in [
                        MONO_SPECIES_SHUFFLE,
                        PURE_PERMUTATION,
                    ] and batch.shuffle_fraction[j] not in [
                        MONO_SPECIES_SHUFFLE,
                        PURE_PERMUTATION,
                    ]:
                        # Alex: Symmetrize since for example 0 and 1 should mean the same
                        true_shuffle_frac_i = min(
                            batch.shuffle_fraction[i], 1 - batch.shuffle_fraction[i]
                        )
                        true_shuffle_frac_j = min(
                            batch.shuffle_fraction[j], 1 - batch.shuffle_fraction[j]
                        )

                        if true_shuffle_frac_i > true_shuffle_frac_j:
                            rank_C_pairs.append((i_nodes, j_nodes))
                        elif true_shuffle_frac_i < true_shuffle_frac_j:
                            rank_C_pairs.append((j_nodes, i_nodes))

                # Fixed chem perturbation
                if same_chem and not same_struct:
                    struct_pairs.append((i_nodes, j_nodes))

                    if batch.sigma[i] > batch.sigma[j]:
                        rank_S_pairs.append((i_nodes, j_nodes))
                    elif batch.sigma[i] < batch.sigma[j]:
                        rank_S_pairs.append((j_nodes, i_nodes))

    return {
        "chem_pairs": chem_pairs,
        "struct_pairs": struct_pairs,
        "rank_S_pairs": rank_S_pairs,
        "rank_C_pairs": rank_C_pairs,
    }


def batch_invariance_loss(
    embeddings: torch.Tensor,
    node_pairs: list[tuple[torch.Tensor, torch.Tensor]],
):
    """
    Compute invariance loss over graph-level embeddings.

    embeddings: shape (num_graphs, embed_dim)
    node_pairs: list of (indices a, indices b)
    """

    if len(node_pairs) == 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    total_loss = 0

    for i_nodes, j_nodes in node_pairs:
        emb_i = embeddings[i_nodes]
        emb_j = embeddings[j_nodes]

        total_loss += nn.MSELoss()(emb_i, emb_j)

    # every node is weighted equally
    return total_loss / len(node_pairs)


def bravais_classification_loss(
    y_bravais: torch.Tensor, batch: torch_geometric.data.Batch
):
    # y_bravais: (total_num_graphs, num_classes)
    # batch.batch: (total_num_nodes,) with graph indices
    # batch.bravais_label: (total_num_graphs,) with class indices

    # extracts node-level labels with batch.bravais_label[batch.batch]
    return nn.CrossEntropyLoss()(y_bravais, batch.bravais_label[batch.batch])


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        lambda_rank: float = 10.0,
        lambda_rep_C: float = 10.0,
        lambda_cls_S: float = 1.0,
        rank_margin: float = 0.1,
        vicreg_gamma: float = 1.0,
    ):
        super(TotalLoss, self).__init__()

        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_rep_C = lambda_rep_C
        self.lambda_rank = lambda_rank
        self.lambda_cls_S = lambda_cls_S
        self.rank_margin = rank_margin
        self.vicreg_gamma = vicreg_gamma

        self.weights = {
            "var_S": lambda_var,
            "var_C": lambda_var,
            "cov_S": lambda_cov,
            "cov_C": lambda_cov,
            "rep_C": lambda_rep_C,
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
            "rep_C": chem_pair_repulsion(
                outputs["z_C"], pairs["chem_pairs"]
            ),  # Alex: combat chemical collapse
            "rank_S": ranking_loss(outputs["s_hat"], pairs["rank_S_pairs"]),
            "rank_C": ranking_loss(outputs["c_hat"], pairs["rank_C_pairs"]),
            "inv_S": batch_invariance_loss(outputs["z_S"], pairs["chem_pairs"]),
            "inv_C": batch_invariance_loss(outputs["z_C"], pairs["struct_pairs"]),
        }
        total = sum(self.weights[k] * v for k, v in loss_dict.items())
        return total, loss_dict
